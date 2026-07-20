# katgpt-rs: Overview

## What It Is

A from-scratch Rust implementation of a GPT-2 style transformer with speculative decoding, designed as an educational/performance research vehicle. No ML frameworks — just `Vec<f32>`, matmul, and hand-tuned attention kernels.

## Project Goals

- CPU-first inference engine with zero-allocation hot paths
- Speculative decoding pipeline (DDTree + DFlash + Leviathan verification)
- Domain-specific constraint pruning (Sudoku, Rust AST via Validator)
- BPE tokenizer + SynPruner for Rust syntax validation
- Sub-millisecond inference on Apple Silicon
- Discrete Diffusion Forcing (dLLM) research with block-parallel denoising

## Current Capabilities

- Single-token autoregressive generation: ~900K tok/s (micro config)
- DFlash marginal prediction: ~4.2M tok/s
- DDTree build: ~431K trees/s
- Speculative decoding: ~1.64M tok/s (AR Draft)
- forward_raven (16 slots): ~1.6M trees/s
- raven_recall (1000 noise): ~9.3M tok/s
- SIMD-accelerated matmul/HLA kernels: 15.6M ops/s [16×16] NEON (Plan 060)
- forward_hla: ~939K tok/s (single-core, 30K CCU feasible)
- forward_ahla: ~1.2M tok/s (single-core)
- TurboQuant 3-bit KV cache: 5.3× compression, 0.99 attention correlation (legacy baseline)
- OCTOPUS octahedral triplet KV cache: 12.2× compression, 0.9512 cosine at 2-bit, -22% to -49% MSE vs SQ — primary KV compression, zero calibration (Plan 099, default-on)
- SpectralQuant calibrated KV cache: 9.1× compression, 0.9917 cosine — secondary KV compression, per-dimension water-fill (Plan 077, default-on)
- ELF SDE noise injection: 10-22× path diversity, logit-normal schedule (Plan 079, default-on)
- CNA Steering: contrastive neuron attribution + sparse modulation, GOAT proved (Plan 087, default-on)
- Deep Manifold: L2/KL residual fixed-point scoring, GOAT 6/6 (Plan 085, default-on)
- Federation: symmetric KL boundary alignment between experts (Plan 085, default-on)
- dLLM Discrete Diffusion Forcing: block-parallel denoising (behind `"dllm"` feature, Plan 066)
- SP-KV self-pruned KV attention: 3-10× KV reduction with utility prediction (behind `"sp_kv"` feature, Plan 070)
- PFlash block-sparse prefill: up to 21.3× sequence reduction, 100% NIAH retrieval
- MaxSim late-interaction scoring: 7.46× SIMD speedup (behind `"maxsim"` feature, Plan 080)
- SimpleTES RPUCG loop: wide>narrow budget scaling (behind `"tes_loop"` feature, Plan 086)
- GDN2 Gated DeltaNet-2: O(1) recurrent attention with decoupled erase/write gates (Plan 105, GOAT 14/14, default-on)
- DashAttention: adaptive sparse hierarchical attention via α-entmax routing (Plan 106, GOAT 9/9, default-on)
- Auto-Dreamer: offline memory consolidation with cadence scheduler + Q-value clustering (Plan 107, GOAT 8/8, default-on)
- LT2 Looped Inference: weight-shared T-pass loop with hybrid SDPA+AHLA dispatch (Plan 108, GOAT 8/8, default-on)
- DMax Soft Parallel Decode: hybrid token/mask embeddings with contiguous prefix promotion (Plan 109, GOAT 7/7, default-on)
- EqR Convergence Selection: Top1Converged picks smallest marginal-change residual (Plan 119, default-on)
- Subterranean Procedure Compilation: user-defined token-rewriting procedures compiled to zero-cost native code (Plan 110, default-on)
- SR²AM Configurator Bandit: per-turn planning regulation via UCB1 (Plan 112, default-on)
- Data Gate: self-play stability via task-level filtering (Plan 111, default-on)
- Plasma Path: ternary SIMD matvec with bit-plane ternary weights, GOAT 5/5 (Plan 117, default-on)
- Parallel-Probe 2D: consensus-based parallel branch control for N branches, GOAT 7/7 (Plan 133, default-on)
- Training-Free Loop: ODE-motivated damped sub-stepping for inference-time refinement, GOAT 4/4 (Plan 136, default-on)
- Newton-Schulz Orthogonalization: 5-iteration cubic fixed-point for Muon-family optimizer weight matrices, GOAT 25/25 (Plan 152, default-on)
- River-Valley Diagnostics: subspace ratios, effective rank, update cosine similarity for convergence analysis, GOAT 25/25 (Plan 152, default-on)
- Sleep Consolidation: offline recursive memory consolidation at KV eviction into GDN2 fast weights, GOAT 14/14 (Plan 154, default-on)
- Spectral Hierarchy: eigenspace alignment + Haar wavelets + Cauchy interlacing for KG extraction validation (Plan 156, default-on)
- Roofline Cost Model: GPU operator runtime prediction via calibrated peak throughput, ~5µs CPU estimate (Plan 159, default-on)
- Tiled Attention: tiled online-softmax flash attention for CPU SIMD (Plan 115)
- Parallax Attention: streaming covariance-corrected local linear attention (Plan 135, opt-in)
- CODA Fusion: fused SIMD kernels matmul+residual+rmsnorm+activation (Plan 103)
- MoA Inference: token-adaptive Mixture-of-Activations SwiGLU over 7-activation dictionary (Plan 158, default-on, GOAT)
- LEO All-Goals: goal-conditioned Q-value trait framework — LeoHead + vectorized Bellman (Plan 155, default-on, SUPER GOAT)
- Dual LEO: teacher-student Q-value mixing + autocurriculum sampling (Plan 155, default-on, SUPER GOAT)
- Sigmoid Margin: SigLIP-style softplus margin loss + dimension sufficiency bound (Plan 157, default-on, GOAT 7/7)
- Kog CPU Fusion: RMSNorm gamma folding + QKV interleaving for monokernel throughput (Plan 160, opt-in)
- Hybrid OCT+PQ: default KV codec — OCT triplet + PlanarQuant 2D Givens rotation (Plan 101, default-on)
- FlashAR Consensus: dual-path ternary thermal routing for consensus tri-mode (Plan 166, GOAT 9/9, default-on)
- Budget Adaptation: compression-adaptive decode budget scaling (Plan 167, GOAT 8/8, default-on)
- Hydra Budget: emergent self-repair layer skipping (Plan 165, GOAT 4/4, default-on)
- GEPA-D Reflective: Pareto bandit config evolution (Plan 164, GOAT 4/4, default-on)
- PhraseBoost: context trie phrase boosting for DDTree (Plan 164, GOAT 5/5, default-on)
- 740+ tests passing (245 test files), 178 examples across 25 groups
- Shared `katgpt-core` crate: types (Config, enums, math utilities), SIMD kernels — extracted for multi-crate reuse
- `QwenDeltaNet` model architecture: hybrid DeltaNet/Attention per-layer config (Plan 182)
- AND-OR DDTree decomposition: relevance-signal hierarchical goal decomposition with memoized subgoals (Plan 190)
- MUX superposition tree search: MuxSpanPruner + MuxDdTree + MuxBfs + mux_demux verifier + MuxBanditWidth arm selector (mux_pruner, mux_ddtree, mux_bfs, mux_demux features)
- LinOSS + ModalSpec drafter: oscillatory state-space cell + Fourier modal speculative drafting (modal_spec feature)
- RiM reasoning buffer slots: K×M reasoning blocks prepended to input, zero-cost slot reuse (rim_slots feature, Plan 172)
- Wall attention: W_g gate projection per KV head dimension, sigmoid-gated attention bypass (wall_attention feature, Plan 173)
- ManifoldPruner: ManifoldE point-to-manifold soft validity scoring + kernel-tricked relevance (behind `"manifold_pruner"` feature, Plan 234, GOAT G1 FAIL — demoted, opt-in only)
- `traits.rs` module in katgpt-core: GameState, RolloutPolicy, StateHeuristic, ActionSpaceLog, ConstraintPruner, ScreeningPruner, SpeculativeGenerator, CollapseDetector, DominoPruner, CompletionHorizon, PartialScorer, ProblemMutator, BestBuddyAligner, DataGate, LeoHead, DualLeoMixer, AutocurriculumSampler, GenerativeConstraintPruner traits
- Sense Composition: KG Latent Octree NPC sense modules with ternary bit-plane projection, GM override dispatch, lock-free hot-swap, and bandit-quality feedback (Plan 221)
- SLoD Spectral Level-of-Detail Pruner: Poincaré ball hyperbolic geometry + heat diffusion on kNN Laplacians for multi-scale KG resolution control (Plan 235, default-ON, GOAT G1–G6)
- Schema Centroid: Per-class embedding centroids for informed KG entity initialization with controlled perturbation (Plan 237, default-ON, GOAT 7/7)
- BAKE Precision-Gated Bayesian Embedding: Per-dimension precision tracking for KG embeddings with O(8) arithmetic, zero-alloc (Plan 236, opt-in, GOAT 10/10 but marginal drift 4.7%)
- Shard Embedding: Johnson-Lindenstrauss random orthogonal projection [f32;64]→[f32;8] for O(1) cosine similarity shard lookup (Plan 230) — **🪦 DEPRECATED (Issue 139):** m=8 violates JL lower bound 200×; marked `#[deprecated]`, zero runtime consumers
- NFCoT FlowScore Drafter: Inference-time normalizing flow density scoring for speculative candidates, zero training (Plan 229)
- Union Bound Confidence: Additive branch confidence via Boole's inequality (Plan 231, default-ON, GOAT 6/6)
- PathwayTracker: Intrinsic pathway stability detection (Plan 231, default-ON, GOAT 7/7)
- FederationComposer: Explicit pruning with residual early termination (Plan 231, default-ON, GOAT 7/7)
- Closed-Unit Compaction Gate (CUCG): generic rubric-gated trajectory compaction primitive (SelfCompact, arxiv 2606.23525) — fires at structurally-safe moments instead of fixed token thresholds. evaluate() 8.91ns, 112.9M/s. Super-GOAT: trajectory compaction and shard freeze are the same primitive (G7). Default-on (Plan 333, 2026-06-25).

## Module Structure

> **Snapshot caveat (2026-07-04):** the per-module listing below is a frozen
> pre-Phase-7 snapshot. The canonical crate layout lives in
> [`README.md` § Crate Dependency DAG](../README.md#crate-dependency-dag) and
> the migration history lives in
[`.proposals/003_src_consolidation_master.md`](../../.proposals/003_src_consolidation_master.md)
> (Phases 0–11 DONE; Phase 12 final sweep pending). Notable moves not reflected
> below: Phase 8 (`closure_wire`, `screening` → `katgpt-pruners`; `rerank` →
> `katgpt-attn-match`), Phase 9 (`mbu`, `tf_loop`, `dense_mesh`, `swir` →
> `katgpt-transformer`), Phase 10 (`cce`, `salience`, `trigger_gate`,
> `skill_opt`, `ssd_block`, `cumprodsum`, `alloc`, `llmexec_guard`,
> `memory_soup_lora`, `mux_demux`, `channel_simd` → `katgpt-core`), Phase 11
> (5 new crates: `katgpt-band`, `katgpt-validator`, `katgpt-sparse`,
> `katgpt-claim`, `katgpt-ruliology`). All moves preserve `katgpt_rs::*`
> import paths via root re-export shims.

```
crates/
  katgpt-core/    Shared types + SIMD kernels (multi-crate reuse):
    types.rs        Config (all presets + with_overrides + validate), Rng, HlaMode, AttentionMode (Causal/Bidirectional/BlockCausal/SpKv/SpKvQuant/DashAttn), ModelArchitecture (Generic/Gemma2/Llama/QwenDeltaNet), WeightDtype (F32/F16/BF16), InferenceOverrides, InferenceResult, DashAttnConfig, DeltaRoutingConfig, DeltaRoutingMode, ConvergenceSelector, LoopMode, HybridPattern, SdpaOutputGate, ResidualGate, PlanningDecision, ConfiguratorContext, DataGate, GateDecision, ProposerTask, TaskType, kv_dim, softmax, softmax_scaled, rmsnorm, rmsnorm_with_gamma, rmsnorm_with_gamma_eps, gegelu, gegelu_tanh, matmul, matmul_relu, sparse_matmul, sample_token, LoraAdapter, LoraPair, DomainLatent
    simd.rs         SimdLevel (Scalar/Neon/Avx2), simd_level(), simd_dot_f32, simd_dot_f16_f32, simd_fma_row, simd_outer_product_acc, simd_matvec, simd_matmul_rows, simd_matmul_rows_parallel, simd_matmul_relu_rows, simd_matmul_f16_f32_rows, simd_matmul_f16_f32_rows_parallel, simd_sparse_dot_f32, simd_sparse_matmul_rows, simd_scale_inplace, simd_fused_decay_write, simd_scale_mul_inplace, simd_exp_inplace, maxsim_score, maxsim_score_packed
    lib.rs          Feature gates: tiled_attention, coda_fusion, parallax_attn, leo_all_goals, dual_leo, questbench, tf_loop, plasma_path, peira_distill, dirichlet_energy, spectral_hierarchy, sigmoid_margin, dual_gram_pca, roofline_cost, domain_latent, sr2am_configurator, data_gate, sparse_mlp, modal_spec, mux_pruner, and_or_dtree, rim_slots, wall_attention, cgsp, action_bridge, sense_composition, slod, spectral_pruner, merkle_octree, flow_field_nav, dec_operators, gpart_adapter, dendritic_gate, qgf_oracle, qgf_projector, qgf_drafter, qgf_adaptive
    traits.rs       ConstraintPruner, DominoPruner, CompletionHorizon, ScreeningPruner, CollapseDetector, GameState, StateHeuristic, RolloutPolicy, LeoHead, AllGoalsUpdate, DualLeoMixer, AutocurriculumSampler, SpeculativeGenerator, GenerativeConstraintPruner, QGradientOracle, PartialScorer, ProblemMutator, BestBuddyAligner + NoPruner, NoScreeningPruner, BinaryScreeningPruner, RandomRolloutPolicy, ActionSpaceLog, GameTrace (Plan 107 Phase 0, consolidated from both crates)
    induced_cwm/    Induced CWM kernel primitive — InducedCwmKernel: GameState marker + CwmCommitment (BLAKE3) + BeliefInferenceFn<S> + TransitionUnitTest + ismcts_search_with_inference + ValueFnTournament + InducedCwmSlot (Plan 296, Research 275, arxiv 2510.04542) ⎗
    attention.rs    Tiled online-softmax flash attention for CPU SIMD (Plan 115, behind "tiled_attention" feature)
    coda.rs         CODA fused SIMD kernels: simd_matmul_rmsnorm_swiglu, simd_matmul_residual, simd_matmul_rmsnorm_rope, simd_matmul_rmsnorm_activation, GateActivation (Plan 103, behind "coda_fusion" feature)
    peira.rs        PEIRA inter-view regressor alignment — EMA cross-view/within-view covariance, closed-form predictor (Plan 153, behind "peira_distill" feature) ⚛
    dirichlet.rs    Dirichlet Energy structural alignment diagnostic — E(E) = Σ A_ij ‖h_i − h_j‖² (Research 111, behind "dirichlet_energy" feature)
    spectral_hierarchy.rs  Spectral hierarchy diagnostic — eigenspace alignment, Haar wavelets, Cauchy interlacing (Plan 156, behind "spectral_hierarchy" feature) ⊕
    questbench.rs   QuestBench underspecification scoring — normalized entropy from ScreeningPruner relevance (Plan 110)
    roofline.rs     Roofline cost model — GPU operator runtime prediction via calibrated peak throughput (Plan 159, behind "roofline_cost" feature) ⊏
    parallax_attn.rs Parallax parameterized local linear attention — streaming covariance correction (Plan 135, behind "parallax_attn" feature) ⊔
    linoss.rs        LinOSS oscillatory state-space cell + ModalSpec drafter — Fourier modal speculative drafting (behind "modal_spec" feature)
    sense/             KG Latent Octree Sense Composition — NPC sense modules with ternary bit-plane projection (Plan 221, behind "sense_composition" feature)
      brain.rs          NpcBrain composition + GM override + HLA projection
      octree.rs         SenseOctreeBuilder — KG→bit-plane octree builder
      gm.rs             GM action dispatch API (pin_sense, disable_autonomous, inject_kg)
      hotswap.rs        SenseHotSwap — lock-free AtomicPtr module replacement
      bandit.rs         SenseTrialLog — bandit trial log + decay direction EMA
      batch.rs          SenseBatch — parallel batch projection (rayon when N>64)
      serialize.rs      SNSE binary format with BLAKE3 verification
      bake.rs           BAKE precision-gated Bayesian embedding update (behind "bake_precision" feature)
      schema_centroid.rs  SchemaCentroidCache — per-class centroid init (behind "schema_centroid" feature)
    shard_embedding.rs  🪦 DEPRECATED (Issue 139): JL random orthogonal projection [f32;64]→[f32;8] — violates JL bound at m=8, marked #[deprecated] (Plan 230)
    slod.rs             SLoD Spectral Level-of-Detail Pruner — Poincaré ball + heat diffusion + tier routing (Plan 235, behind "slod" feature)
    and_or/          AND-OR tree module — AndOrNode<G,S> generic AND-OR tree for hierarchical goal decomposition (behind "and_or_dtree" feature)
      mod.rs        Module root, re-exports AndOrNode
      types.rs      AndOrNode enum (Or/And/Leaf), is_solved, push_child, set_best, set_solution
    mux/             MUX superposition tree search — superposition DD-tree with BFS frontier (behind "mux_pruner" feature)
      mod.rs        Module root — mux_pruner, mux_ddtree, mux_bfs, mux_demux, mux_bandit_width sub-features
      span_pruner.rs  MuxSpanPruner — superposition span validation
      top_k.rs      extract_top_k_peaks — top-K peak extraction from logit distributions
      dd_tree.rs    MuxDdTree, MuxNode — superposition DD-tree with hypothesis coverage
      bfs.rs        MuxBfs — dynamic-width BFS frontier expansion
      demux.rs      mux_demux — deterministic superposition recovery verifier
      bandit_width.rs  MuxBanditWidth — UCB1 arm selector for tree width
      freeze_thaw.rs   MuxTarget, MuxPatternStore — freeze/thaw for superposition patterns

src/
  lib.rs            Module index + debug tracking allocator
  main.rs           Entry point (proof → bench → Percepta bench → plot)
  types.rs          Re-exports katgpt_core::types::* (including DashAttnConfig, DeltaRoutingConfig, ConvergenceSelector, LoopMode, HybridPattern, SdpaOutputGate, ResidualGate, PlanningDecision, ConfiguratorContext, DataGate, GateDecision, ProposerTask, TaskType) + QuantizedKVCache trait (interface for TurboQuant/SpectralQuant KV caches)
  simd.rs          SimdLevel (Scalar/Neon/Avx2), simd_level(), simd_dot_f32, simd_fma_row, simd_outer_product_acc, simd_matvec, simd_matmul_rows, simd_matmul_relu_rows, simd_sparse_dot_f32, simd_sparse_matmul_rows, simd_scale_inplace (Plan 060)
  transformer.rs    TransformerWeights (+ mtp projections), LayerWeights, KVCache, MultiLayerKVCache, KVSnapshot, PagedKVCache, RavenKVCache, ForwardContext (+ sparse buffers + lora_buf + mtp_context_buf + tq_dequant_pos), PrefillContext, DecodeStage, forward, forward_with_domain_latent, forward_prefill, forward_paged, forward_raven, forward_turboquant, forward_looped, forward_coda, forward_decode_stage, depth_route_weights, generate, generate_into, generate_batch, generate_with_prefill, tokens_to_string, project_target_activation, cluster_map_round_robin, cluster_map_from_embeddings, raven_compute_router, raven_update, raven_readout, preload_kv_cache
  weights.rs        ContiguousWeights — single-buffer 64-byte aligned weight layout (Plan 102)
  feedback.rs       FeedbackConfig, send_feedback ⌁
  percepta/         Percepta 2D Convex Hull Attention + Computation Graph:
    mod.rs          Module declarations, re-exports (15+ submodules)
    types.rs        TieBreak, HullMeta, Vec2 (f64), constants (HARD_K, BIG, EPS)
    legacy.rs       Vec2 (f32), KVCache2D (Graham Scan), Sudoku9x9, SymbolicValidator, StreamingSolver, SolveEvent
    cht.rs          Line, CHT — dynamic convex hull trick / line container
    hull.rs         AttentionResult, HullHalf, HardAttentionHead (dual-hull O(log N)), BruteAttentionHead
    encoding.rs     encode_key, encode_query, clear_key, hard_scale, hard_scale_query
    cumsum.rs       CumSum — cumulative sum via uniform attention
    standard_cache.rs  StandardCache — O(n) softmax attention KV cache
    gates.rs        reglu, stepglu, multiply — gate primitives; PersistSlot, GateKind
    graph/          Computation Graph DSL:
      mod.rs        Module root, re-exports
      types.rs      Expression (sparse linear combo), DimensionKind, Dimension, LookUp, ProgramGraph, GraphBuilder, ValidationError
    weights.rs      TransformerWeights, LayerWeights, AttentionWeights, FfnWeights, HeadInfo, build_weights
    transformer.rs  TransformerConfig, TransformerVocab, GenerationResult, VanillaTransformer
    evaluator.rs    GraphEvaluator — step/predict/evaluate/compare_with_reference
    specialize.rs   SpecializationError, SpecializationReduction, SpecializedModel, UniversalModel
    scheduler.rs    OpKey, Phase, StdLayer, DepGraph, Schedule, build_dep_graph, milp_schedule
    runner.rs       RunnerError, BuildResult, Runner — compile/build/run/evaluate/specialize/full_pipeline
    compile.rs      compile_program, CompiledProgram — C source → WASM → lowered bytecode → token prefix (behind "percepta_compile")
    wasm/           WASM MVP decoder + lowering + interpreter (Futamura projection):
      mod.rs        Module root
      decoder.rs    WasmModule, FuncType, FuncBody, WasmInstr, decode
      lower.rs      lower_hard_ops, check_basic_only
      interpreter/  WASM interpreter as computation graph:
        mod.rs      Module root
        arithmetic.rs  Arithmetic ops dispatch
        dispatch.rs    Instruction dispatch table
        tokens.rs      Token mapping
  tf_loop.rs        Training-Free Loop — ODE-motivated damped sub-stepping inference-time refinement (Plan 136) ⊛---
  newton_schulz.rs  Newton-Schulz orthogonalization + Muon momentum — 5-iteration cubic fixed-point (Plan 152) ☊
  river_valley.rs   River-valley diagnostic metrics — subspace ratios, effective rank, update cosine similarity (Plan 152) ☊
  ega_attn.rs       Energy-Gated Attention — spectral salience gating with z-normalized sigmoid gate (Plan 139) ⍰
  shard_kv/         ShardKV asymmetric K/V compression (Plan 147) ⎘:
    mod.rs          Module root (re-exports)
    types.rs        ShardKV layer + config types
    rope.rs         RoPE undo for PCA rotation path
    kv_cache.rs     ShardKV KV cache impl (K: PCA+water-fill, V: Hadamard+K-means)
  sleep/            Sleep Consolidation — offline recursive memory consolidation at eviction (Plan 154) ☽:
    mod.rs          Module root, re-exports
    types.rs        SleepConfig, SleepLayer, SleepSnapshot
    consolidation.rs N-pass offline recurrent consolidation into GDN2 fast weights
    eviction.rs     KV cache eviction + consolidation pipeline
  distill/          PEIRA distillation (Plan 153) ⚛:
    mod.rs          Module root (behind "peira_distill" feature)
    peira.rs        PEIRA inter-view regressor alignment — collapse-free modelless distillation
    ilc.rs           ILC (Iterative Latent Clustering) Distillation — synonym-aware DDTree pruning (behind "ilc_distill" feature) ⚛+
  benchmark.rs      BenchCategory, BenchResult, run_all, run_all_parallel, save_results_csv, append_timeseries_csv, generate_batch, bench_hla_vs_flat_cache, bench_hla_memory, bench_hla_quality, bench_simd, bench_sparse_mlp
  plot.rs           plot_results → PNG, plot_timeseries
  rerank.rs         RerankMethod (Cosine/MaxSim), RerankedDoc, ndcg_at, SymmetricBoundaryPair (behind "maxsim" + "bt_rank" features)

  speculative/      SOLID decomposition:
    mod.rs          Re-exports
    types.rs        TreeNode, DraftResult, ConstraintPruner trait, ScreeningPruner trait, NoPruner, NoScreeningPruner, BinaryScreeningPruner, SpeculativeContext, DDTreeBranchCache, RejectionReason, DraftEvent, PrefillMode, FlashPrefillConfig, BlockScores
    sampling.rs     sample_from_distribution, sample_residual_distribution, sample_residual_distribution_into
    dd_tree.rs      build_dd_tree, build_dd_tree_pruned, build_dd_tree_screened, build_dd_tree_screened_with_schedule (thinking_prune), build_dd_tree_balanced, TreeBuilder, extract_parent_tokens, extract_parent_tokens_into, extract_best_path, extract_best_path_into, build_inference_result, merge_retrieved_branches
    dflash.rs       dflash_predict, dflash_predict_with, dflash_predict_ar, dflash_predict_ar_with, dflash_predict_conditioned, dflash_predict_conditioned_with, dflash_predict_parallel
    verifier.rs     SpeculativeVerifier trait, SimulatedVerifier, LeviathanVerifier
    step.rs         speculative_step, speculative_step_verifier, speculative_step_rollback, speculative_step_rollback_with, speculative_step_conditioned, speculative_step_conditioned_with, speculative_step_rollback_paged
    prefill.rs      PrefillScorer trait, AttentionScorer, BlockAttentionScorer, compress_prompt, compress_prompt_blocks, block_select, block_select_grid, should_compress, speculative_prefill, speculative_prefill_block, speculative_prefill_adaptive
    flow_pruner.rs  FlowPruner<P> — GFlowNet-inspired stop-probability regularization ♭
    d2f_verifier.rs D2fDrafterVerifier — D2F diffusion drafts, AR verifies (Tri-Mode, Plan 089) ⓘ
    d2f.rs          D2fBlockState, D2fDecodeConfig, D2fBlockResult, D2fPipelineBlock, D2fPipeline, D2fPipelineResult, d2f_decode_block* (behind "dllm" feature)
    alpha.rs        AlphaTarget, alpha_intersect, is_consistent — LDT α-intersection pruning + conflict detection (behind "lattice_deduction" feature, Plan 088) ⎌
    ppot/           PPoT (Plans 026 + 027) ○
      mod.rs        Module root, public API re-exports
      types.rs      TokenRule enum, PpotConfig
      entropy.rs    token_entropy, identify_high_entropy_positions, identify_positions_by_rule, identify_positions_adaptive
      resample.rs   ppot_resample, ppot_resample_with_support, ppot_resample_different_value, ppot_resample_multi_strategy, ppot_rescue, ppot_rescue_adaptive, ppot_rescue_reviewed
      knowledge.rs  RejectionInsight, ErrorKind, SessionKnowledge
      rank.rs       rank_by_consistency, rank_by_consistency_weighted, select_best_variant, select_best_variant_weighted
      flashar_anchor.rs  FlashAR Strided Anchor-Then-Fill D2F Decoding (Plan 166 T11, behind "flashar_anchor" feature) ⚓
      flashar_consensus.rs  FlashAR Consensus Tri-Mode with Ternary Thermal Paths (Plan 166, behind "flashar_consensus" feature) ⚖
      budget.rs        Compression-adaptive decode budget (Plan 167, behind "budget_adaptation" feature) 💰
      budget_compat.rs  Budget adaptation integration helpers (Plan 167 Phase 2)

  pruners/          Domain-specific constraint pruners:
    mod.rs          Re-exports
    pathfinder.rs   Target, find_path, find_distance, reachable_positions, enumerate_targets, terrain_cost, manhattan
    tactical_pruner.rs  GameState, TacticalPruner (grid-based tactical puzzle)
    dungeon_pruner.rs   FloorGrid, StairConnection, DungeonMap, DungeonState, DungeonPruner (multi-floor)
    dungeon_pathfinder.rs  DungeonAction, MultiFloorTarget, find_path_on_floor, find_path_multifloor, enumerate_multifloor_targets
    map_generator.rs  GeneratedMap, GeneratedDungeon, MapGenerator (procedural generation)
    sudoku_pruner.rs  SudokuPruner *
    bandit.rs       BanditStrategy, BanditStats, BanditPruner<P>, BanditSession, BanditEvent, BanditResult, BanditEnv trait, BernoulliEnv, GaussianEnv, SharedBanditStats ♭
    trial_log.rs    TrialRecord, TrialSummary, TrialLog ♭
    absorb_compress.rs  CompressConfig, AbsorbCompress trait, AbsorbCompressLayer<P> ♭
    hot_swap.rs     HotSwapPruner<P> — blake3 hash comparison reload ♭
    regression.rs   GoldenTrace, RegressionFailure, RegressionResult, RegressionSuite, ReplayReward trait ♭
    review_metrics.rs  ReviewSummary, ReviewMetrics, ReviewStrategy, EntropyAnomalySummary ♭
    stepcode.rs     PathStep, ShapedPath, shape_path, path_consistency ≋
    variance_minimizer.rs  VarianceMinimizer, VarianceMinimizerConfig (Plan 078) ☀
    freeze.rs       save_frozen, load_frozen — shared freeze/thaw disk I/O for repr(C) bandit knowledge structs (Plan 092)
    game_state/     GameState forward model trait + generic MCTS (Plan 056) ⎗
      mcts_search   mcts_search — Monte Carlo Tree Search
                    StateHeuristic trait, ActionSpaceLog
    bomber/         Bomberman 4-player HL arena (bevy_ecs) ⍟
      mod.rs        BomberAction, PowerUpKind, Cell, ECS components/resources, GameEvent
      arena.rs      ArenaGrid — 13×13 grid generation + presets
      players.rs    BomberPlayer trait, RandomPlayer, GreedyPlayer, ValidatorPlayer, HLPlayer, LoraPlayer, LoraWasmPlayer, NNPlayer
      g_zero_player.rs  GZeroPlayer — G-Zero self-play with template proposer + delta bandit
      tft_player.rs  TftPlayer — Tit-for-Tat with provocation detection
      rubric_player.rs  RubricPlayer — rubric-vector reward (Plan 071 T9)
      sdar_player.rs  SdarBomberPlayer — SDAR sigmoid-gated reward (Plan 072)
      arena_runner.rs  BomberArenaConfig, run_bomber_game, run_bomber_matchup (Plan 076)
      replay.rs     ReplaySample, ReplayWriter — JSONL replay persistence
      replay_backward.rs  BackwardSample, ReplayBackwardWalker — GFlowNet backward policy
      systems.rs    init_world, spawn_players, run_tick
      wasm_pruner.rs  BomberWasmPruner — WASM batch validation
      wasm_state.rs  serialize_game_state, ZeroCopyStateBuffer
    monopoly/       Monopoly board game engine (bevy_ecs) ✦
      mod.rs        PropertyGroup, SquareKind, TurnPhase, GameEvent (30+ variants), Player, Property, Board, etc.
      board.rs      build_board, shuffle_decks, group_squares
      players.rs    MonopolyPlayer trait, RandomPlayer, GreedyPlayer, ValidatorPlayer, HLPlayer, DecisionContext, Strategy
      systems.rs    init_world, spawn_players, execute_turn, run_game, calculate_rent, transfer_assets
    fft/            FFT Tactics Arena — ATB battle engine ✧
      mod.rs        Module root, re-exports
      types.rs      Class (6), Team, ActionType (9), Stats, Pos, Unit, Action, GameEvent
      battle.rs     BattleState, resolve_action, should_forgive
      status.rs     StatusEffect (9), ActiveEffect, apply_tick_effects, can_cast, can_act, ct_fill_rate
      players.rs    FftPlayer trait, GreedyFFTPlayer, ValidatorFFTPlayer, HLFFTPlayer
      g_zero_player.rs  GZeroFFTPlayer — G-Zero self-play for FFT
      rubric_player.rs  RubricFFTPlayer — rubric-vector reward (Plan 071 T10)
      sdar_player.rs  SdarFFTPlayer — SDAR sigmoid-gated reward (Plan 072)
      arena_runner.rs  FftArenaConfig, run_fft_battle, run_fft_matchup (Plan 076)
      tft_player.rs  TftFFTPlayer — Tit-for-Tat FFT player
    go/             Go GameState + AutoGo API bridge + tournament ⛩
      mod.rs        Module root, re-exports
      types.rs      GoAction (Place, Pass), GoCell (Empty, Black, White)
      state.rs      GoState — flat array board, simple ko, Tromp-Taylor scoring, GameState trait, GoHeuristic
      autogo_client.rs  AutoGoClient — REST API bridge to AutoGo play.py server
      replay.rs     GoReplay, MoveRecord — game recording + deterministic playback
      players.rs    GoPlayer trait, GoRandomPlayer, GoGreedyPlayer, GoValidatorPlayer, GoHLPlayer, GoGZeroPlayer, GoMctsPlayer
      tournament.rs GoTournamentConfig, GoTournamentResult, AutoGoProxyPlayer, run_tournament
      g_zero_player.rs  GoGZeroSelfPlay — HintDelta + absorb-compress self-play
      autoresearch.rs   AutoResearchLoop — UCB1 bandit over config arms, early stopping
      analytics.rs  cross-domain analysis, scaling laws, player tier comparison
    delta_mem/      δ-Mem modelless distillation — associative bandit memory ⌘
      mod.rs        Module root, re-exports
      state.rs      DeltaMemoryConfig, DeltaMemoryState, DeltaMemorySnapshot
      hash.rs       FeatureHasher, ContextFeatures, OutcomeFeatures
      pruner.rs     CorrectionMode, WriteGranularity, MemorySteeredPruner<P>
      multi.rs      AggregationStrategy, MultiDomainMemory
      multi_pruner.rs  MultiDomainMemoryPruner<P>
    g_zero/         G-Zero self-play distillation — verifier-free self-evolution ǂ
      mod.rs        Module root, re-exports
      types.rs      HintDelta, LogProbResult
      template_proposer.rs  QueryTemplate, GeneratedPair, TemplateProposer
      bomber_templates.rs  BomberTemplate (8 strategies), BomberTemplateProposer
      delta_bandit.rs  DeltaBanditPruner<P>
      delta_absorb.rs  DeltaGatedConfig, DeltaGatedAbsorbCompress<P>
      fft_templates.rs  FFTTemplate (10 strategies), FFTTemplateProposer

    dreamer/        Auto-Dreamer offline memory consolidation (Plan 107, behind "dreamer" feature) ∞:
      mod.rs          Module root, re-exports
      types.rs        DreamerConfig, CadenceSchedule, QCluster
      scheduler.rs    cadence scheduler — when to consolidate
      consolidator.rs offline Q-value consolidation pass
      pipeline.rs     DreamerPipeline — full consolidation pipeline
      counterfactual.rs  counterfactual replay generation
      decay.rs        exponential decay for stale memories
      frozen.rs       frozen memory snapshot I/O
    subterranean/   Procedure graph compilation — compiling workflows into weights (Plan 110, behind "subterranean" feature) ≬:
      mod.rs          Module root, re-exports
      types.rs        ProcedureGraph, ProcedureNode, CompiledProcedure
      cost_model.rs   procedure cost estimation
      path_enumerator.rs  enumerate procedure paths
      path_sampler.rs     sample procedure paths
      training_mode.rs    training mode dispatch
      bandit_bridge.rs    bridge to bandit infrastructure
      game_bridge.rs      bridge to game state trait
      bomber_procedure.rs Bomberman procedure definitions
      go_procedure.rs     Go procedure definitions

    arena/           Cross-arena tournament infrastructure (Plan 076):
      mod.rs        Module root + re-exports
      types.rs      ArenaKind, GameResult, MatchupResult, Ranking, Leaderboard, EloCalculator
      scheduler.rs  Matchup, round_robin_pairs, full_field_matchups
    ropd_rubric/     ROPD rubric modelless distillation (Plan 071):
      mod.rs           Module root + re-exports
      template.rs      RubricCriterion, RubricTemplate (bomber/fft/generic)
      types.rs         RubricVector (weighted_score, gap_vs_references)
      scorer.rs        RubricScorer trait, PatternScorer, score_with_references
      rubric_absorb.rs RubricGatedAbsorbCompress<P> (per-criterion gated absorb)
      rubric_bandit.rs RubricBanditPruner<P> (rubric-weighted reward bandit)

    sdar_gate.rs     SDAR sigmoid gate primitives (sdar_gate, sdar_modulate, sdar_gated_reward)
    bt_rank.rs       BtOutcome, BtComparison, BtConfig, BtScores, bt_fit, bt_fit_from_fn, bt_sigmoid — Bradley-Terry pairwise ranking ⊞
    cna.rs           CnaNeuron, CnaCircuit, CnaDiscoveryConfig, CnaModulator, CnaScreeningPruner, cna_discover, cna_modulate — Contrastive Neuron Attribution 🔬
    manifold_residual.rs  KlResidualScorer, L2ResidualScorer, ManifoldResidual, ResidualRelevanceScorer — Deep Manifold fixed-point scoring ∇
    boundary_alignment.rs  BoundaryAlignment trait, KlBoundaryAligner — federated KL coupling ≋
    tes_loop.rs      TesLoop trait, SimpleTesLoop<E>, TrajectoryPruner — SimpleTES RPUCG loop ⟳
    hydra_budget.rs  Hydra-Aware Adaptive Layer Budget (behind "hydra_budget" feature) 🐉
    gepa_reflective.rs  GEPA-D Reflective Config Evolution (behind "gepa_reflective" feature) 🪞
    phrase_boost.rs  PhraseBoost context trie phrase boosting (behind "phrase_boost" feature) 📝
    phrase_trie.rs   Compact token-level trie for phrase boosting (behind "phrase_boost" feature) 🌳

    sdar/            SDAR gated distillation — modelless (Plan 072):
      mod.rs           Module root + re-exports
      sdar_bandit.rs   SdarBanditPruner<P> (sigmoid-gated reward updates)
      sdar_absorb.rs   SdarGatedAbsorbCompress<P> (soft sigmoid promotion)

  tokenizer/        BPE tokenizer (encode/decode/train, Config::bpe())
    mod.rs          Re-exports: BpeTokenizerImpl, BpeTrainer, BpeTokenizer, MergeRule
    types.rs        BpeTokenizer, MergeRule
    bpe.rs          BpeTokenizerImpl (encode/decode), BpeTrainer (train)

  validator/        SynPruner + partial parser ‡
    mod.rs          Module root
    types.rs        PruneResult, ErrorKind, CompilerFeedback
    partial_parser.rs  PartialParser — bracket balance DFA (Tier 0)
    syn_pruner.rs   SynPruner — two-tier pruner (DFA + syn parse)

  turboquant/      TurboQuant KV cache compression — legacy baseline for bench/educate only (arXiv:2504.19874):
    mod.rs          Module root (re-exports)
    types.rs        TurboQuantCodebook, TurboQuantLayer, TurboQuantKVCacheConfig
    codebook.rs     Lloyd-Max codebook (compute_codebook, quantize, dequantize)
    rotation.rs     QR-based orthogonal rotation + QJL projection
    kv_cache.rs     TurboQuantKVCache (store_key, store_value, dequantize, bit-pack)
    forward.rs      attention_turboquant, dequantize_keys_flat/values_flat, cosine_similarity

  octopus/         OCTOPUS octahedral triplet KV compression — primary default (Plan 099) ⊛:
    mod.rs          Module root (re-exports)
    types.rs        OctopusConfig, OctopusLayer, OctopusCodebook, TripletIndices
    octahedral.rs   oct_encode, oct_decode — S² ↔ [-1,1]² equal-area parameterization
    triplet.rs      Triplet, decompose, recompose, recompose_into — 3-block grouping
    codebook.rs     ScalarCodebook, build_norm_codebook, build_oct_codebook — Lloyd-Max codebooks
    encode.rs       encode_triplet, joint_3x3_round, bit-pack/unpack — triplet encoder
    kv_cache.rs     OctopusKVCache — QuantizedKVCache trait impl
    forward.rs      maxsim_score_octopus, dequantize-to-flat — score-path decode (behind "maxsim" feature)

  hybrid_oct_pq/   Hybrid OCT triplet + PlanarQuant rotation — default KV codec (Plan 101) ⊛+:
    mod.rs          Module root (re-exports)
    types.rs        HybridOctPqConfig, HybridOctPqLayer
    kv_cache.rs     HybridOctPqKVCache — QuantizedKVCache trait impl
  planar_quant/    PlanarQuant 2D Givens rotation KV cache (Plan 100, behind "planar_quant" feature) ⊕:
    mod.rs          Module root (re-exports)
    types.rs        PlanarQuantConfig, PlanarQuantLayer
    rotation.rs     2D Givens rotation — O(d) vs TQ O(d²)
    kv_cache.rs     PlanarQuantKVCache — QuantizedKVCache trait impl
  iso_quant/       IsoQuant 4D quaternion rotation KV cache (Plan 100, behind "iso_quant" feature) ⊕+:
    mod.rs          Module root (re-exports)
    types.rs        IsoQuantConfig, IsoQuantLayer
    rotation.rs     4D quaternion rotation — O(d) vs TQ O(d²)
    kv_cache.rs     IsoQuantKVCache — QuantizedKVCache trait impl

  spectralquant/   SpectralQuant calibrated KV compression — secondary, per-dimension water-fill (Plan 077) ⊛:
    mod.rs          Module root (re-exports)
    types.rs        LloydMaxCodebook, SpectralQuantCalibration, WaterfillAllocation, SpectralQuantLayer, SpectralQuantKVCacheConfig
    spectral.rs     calibrate_eigenbasis, waterfill_bits, participation_ratio, spectral_gap, LloydMaxQuantizer
    nonuniform_quant.rs  NonUniformQuantizer, CompressedVector — Lloyd-Max scalar quantizer
    spectral_rotation.rs  SpectralRotation — eigenbasis rotation, RandomRotation (turboquant compat)
    spectral_kv_cache.rs  SpectralQuantKVCache, DequantizeScratch — full quantized KV cache implementation
    forward.rs      attention_spectralquant, dequantize_spectral_keys_flat/values_flat, par_maxsim_score_spectralquant (behind "maxsim" feature)

  dllm.rs          NoiseSchedule, D2fContext, DenoiseConstraint trait, corrupt_block, forward_bidirectional_positions, forward_block_causal_positions, denoise_loop, denoising_accuracy ⌂
  dash_attn/       DashAttention adaptive sparse hierarchical attention (Plan 106, behind "dash_attn" feature) ∹
    mod.rs          Module root, re-exports
    entmax.rs       α-entmax sparse attention activation
    routing.rs      chunk-level routing + importance scoring
    chunk_summary.rs  chunk summary statistics
    forward.rs      forward_dash_attn, forward_dash_attn_with_config
    tests.rs        unit tests
  gdn2/            Gated DeltaNet-2 recurrent attention (Plan 105, behind "gdn2_attention" feature) ◉
    mod.rs          Module root, re-exports
    types.rs        Gdn2Config, Gdn2State, Gdn2Gate
    kernel.rs       simd_fused_decay_write-based recurrent update
    forward.rs      forward_gdn2, forward_gdn2_with_state
  hla/             Higher-order Linear Attention — O(1) inference cache (Plan 057, SIMD Plan 060) ⎔
    mod.rs          Module root
    types.rs        HlaQHeadState, HlaLayerState, MultiLayerHlaCache, AhlaQHeadState, AhlaLayerState, MultiLayerAhlaCache, HlaVariant
    kernel.rs       hla_state_update, hla_readout, hla_denom, ahla_step, ahla_denom, hla_layer_update, hla_layer_readout, ahla_layer_step
    forward.rs      forward_hla, forward_ahla, generate_hla_into, generate_ahla_into
  sp_kv/           Self-Pruned Key-Value Attention (Plan 070) §
    mod.rs          Module root
    types.rs        SpKvGateMode, SpKvConfig, SpKvLayerCache, SpKvCache, UtilityPredictorWeights, SpKvPredictors, GateBiasBuffer
    utility_predictor.rs  predict, predict_single_head, soft_gate_bias, hard_gate_bias, tahg_gate_bias, UtilityAggregation
    forward.rs      SpKvForwardContext, BiasProvider trait, attention_head_core, attention_head_gated, forward_sp_kv

  unit_distance/    Unit Distance GOAT proof — number-theoretic lattice constructions (Plan 090, behind "unit_distance" feature) 📏:
    mod.rs          Module root, re-exports
    types.rs        LatticePoint, DistanceProof
    cm_field.rs     CM-field constructions
    minkowski.rs    Minkowski bound computations
    pigeonhole.rs   Pigeonhole principle proofs

  data_probe/      Data Probe Diagnostics — information-theoretic validation (Plan 141, behind "data_probe" feature) 🔍:
    mod.rs          Module root
    markov.rs       Dirichlet-sampled Markov chain generator
    nll.rs          NLL computation against known chain
    typical_set.rs  Three-way regime classification
    dirichlet_energy.rs  Dirichlet Energy structural alignment diagnostic
    claim.rs        Claim card infrastructure for C1-C4 validation
    geometry.rs     Representation geometry diagnostics (Plan 151)
  skill_opt/       SkillOpt text-space skill optimization (Plan 144, behind "skill_opt" feature) ✎:
    mod.rs          Module root
    edit.rs         Edit operations and SkillEdit struct
    apply.rs        Deterministic text patching engine
    gate.rs         Validation gate
    schedule.rs     Edit budget schedules
    buffer.rs       FIFO ring buffer for rejected edits
    optimizer.rs    SkillOptimizer trait
  proof_cert/      Hierarchical GOAT Proof Certificates (Plan 145, behind "proof_cert" feature) 🏆:
    mod.rs          Module root
    certificate.rs  Certificate types (ProofCertificate, ProofEvidence, ProofProperty, ProofResult)
    chain.rs        Certificate chain verification
    macros.rs       Declarative proof macros
    serde_impls.rs  Serde serialization + checksum
    wasm_certificates.rs  WASM certificate generation
  cache_prune/     CachePrune SAT + rolling hash + sensitivity (Plan 140, behind "cache_prune" feature) ✂:
    mod.rs          Module root
    rolling_hash.rs Rolling hash for O(n) variable-length segment matching
    sat.rs          Summed-Area Table for O(1) rectangular attention queries
    sensitivity.rs  Generic SensitivityDetector trait

  alloc.rs          Debug-only TrackingAllocator, reset_alloc_stats, get_alloc_stats (debug builds)

  * behind --features sudoku
  ∘ behind --features sparse_mlp    (default)
  ○ behind --features ppot           (default)
  ‡ behind --features validator
  ♭ behind --features bandit         (default)
  ⍟ behind --features bomber         (bevy_ecs + bandit)
  ✦ behind --features monopoly       (bevy_ecs + bandit)
  ✧ behind --features fft            (bandit)
  ⛩ behind --features go             (bandit + reqwest)
  ⌘ behind --features delta_mem      (bandit)
  ǂ behind --features g_zero         (bandit)
  ⌁ behind --features feedback
  ⎔ behind --features hla_attention
  § behind --features sp_kv
  ⌂ behind --features dllm
  ≋ behind --features stepcode
  ⎗ behind --features game_state
  ⊛ behind --features spectral_quant  (default)
  ☀ behind --features replaid_schedules
  ⊞ behind --features bt_rank         (default)
  ⊘ behind --features sdar_gate
  ⊡ behind --features ropd_rubric     (bandit)
  ⚡ behind --features elf_sde         (default)
  🔬 behind --features cna_steering    (default)
  ∇ behind --features deep_manifold    (default)
  ≋ behind --features federation       (default)
  ⟳ behind --features tes_loop         (bandit)
  ⬡ behind --features maxsim
  ▣ behind --features percepta          (ordered-float)
  ▣+ behind --features percepta_gates   (percepta)
  ▣++ behind --features percepta_graph  (percepta_gates)
  ▣+++ behind --features percepta_wasm  (percepta_graph)
  ▣++++ behind --features percepta_compile (percepta_wasm + good_lp)
  ⎌ behind --features lattice_deduction
  ⊛+ behind --features hybrid_oct_pq (default)
  ⊕ behind --features planar_quant
  ⊕+ behind --features iso_quant
  ∹ behind --features dash_attn (default)
  ◎ behind --features mls_aggregate (default)
  ◉ behind --features gdn2_attention (default)
  ∞ behind --features dreamer (default)
  ↻ behind --features lt2_looped (default)
  ⊞+ behind --features dmax_spd (default)
  ERRQ behind --features eqr_convergence (default)
  ≬ behind --features subterranean (default)
  ⚙ behind --features sr2am_configurator (default)
  ⊇ behind --features data_gate (default)
  ◧ behind --features tiled_attention
  ⨍ behind --features coda_fusion
  📏 behind --features unit_distance
  📊 behind --features stability_metrics (default)
  ⎗+ behind --features decode_specialize
  ⓘ behind --features tri_mode (dllm)
  ⊛- behind --features plasma_path   (default)
  ⊛-- behind --features parallel_probe (default)
  ⊛--- behind --features tf_loop      (default)
  ☊ behind --features newton_schulz    (default)
  ☊ behind --features river_valley    (default)
  ⍰ behind --features ega_attn        (opt-in)
  ⎘ behind --features shard_kv        (opt-in)
  ☽ behind --features sleep_consolidation (default)
  ⚛ behind --features peira_distill   (default)
  ⚛+ behind --features ilc_distill
  ⊕ behind --features spectral_hierarchy (default)
  ⊏ behind --features roofline_cost    (default)
  ⊔ behind --features parallax_attn   (opt-in)
  ⚓ behind --features flashar_anchor    (dllm)
  ⚖ behind --features flashar_consensus (tri_mode, plasma_path)
  💰 behind --features budget_adaptation
  🐉 behind --features hydra_budget     (default)
  🪞 behind --features gepa_reflective  (bandit, memo_reflections, default)
  📝 behind --features phrase_boost     (default)
  Plans 137-145 modules are opt-in, see Feature Flags table
```

## Feature Flags

| Flag | Dependencies | Description |
|------|-------------|-------------|
| `sparse_mlp` | — | TwELL-inspired sparse MLP matmul (Plan 022) |
| `ppot` | — | PPoT logit-parameterized CPU resampling + adaptive rescue (Plans 026 + 027) |
| `domain_latent` | — | Free Transformer mid-layer domain conditioning (Plan 038) |
| `bandit` | — | Multi-armed bandit + HL infrastructure: TrialLog, AbsorbCompress, HotSwapPruner, RegressionSuite, ReviewMetrics (Plans 030–032) |
| `sudoku` | — | SudokuPruner constraint pruning + examples |
| `validator` | `syn`, `proc-macro2` | SynPruner + partial parser |
| `delta_mem` | `bandit` | δ-Mem modelless distillation — associative bandit memory (Plan 053) |
| `g_zero` | `bandit` | G-Zero self-play distillation — Hint-δ gated absorb + bandit (Plan 049) |
| `hla_attention` | — | HLA/AHLA streaming attention kernels (Plan 057, SIMD-accelerated in Plan 060) |
| `fft` | `bandit` | FFT Tactics Arena — ATB battle engine with status effects (Plan 053) |
| `bomber` | `bevy_ecs`, `bandit` | Bomberman HL arena (Plan 033) |
| `bomber-wasm` | `bomber`, `wasmtime`, `papaya` | WASM bomber validator loader + batch pool (Plans 034 + 037) |
| `monopoly` | `bevy_ecs`, `bandit` | Monopoly board game engine (Plan 035) |
| `feedback` | — | E2E feedback loop — sends inference results to REST endpoint (Plan 042) |
| `rest` | — | REST bridge test + merge stub (Plan 009, client lives in riir-ai/riir-rest) |
| `embedding_router` | — | Semantic embedding routing (Plan 024) |
| `game_domain` | `domain_latent` | Alias for domain_latent — game-specific Config presets (Plan 040) |
| `language_domain` | — | Language domain: BPE vocab, LLM models (Plan 040) |
| `gpu` | — | Placeholder — GPU training lives in riir-ai/riir-gpu |
| `go` | `bandit`, `reqwest` | Go GameState + AutoGo API bridge + tournament + G-Zero self-play + AutoResearch loop (Plan 065) |
| `sp_kv` | — | SP-KV self-pruned key-value attention (Plan 070) |
| `dllm` | — | D2F Discrete Diffusion Forcing — mini dLLM research (Plan 066) |
| `stepcode` | `bandit` | Path shaping + consistency scoring (Plan 054, infrastructure only, no perf gain) |
| `ropd_rubric` | `bandit` | ROPD rubric modelless distillation — multi-criteria reward vectors, per-criterion gap targeting (Plan 071) |
| `sdar_gate` | — | SDAR sigmoid-gated distillation — asymmetric trust for bandit updates + soft absorb promotion (Plan 072) |
| `bt_rank` | — | Bradley-Terry pairwise ranking for DDTree selection (OpenDeepThink distillation) |
| `spectral_quant` | — | SpectralQuant calibrated eigenbasis + water-fill bit allocation — secondary KV compression, useful for per-dimension water-fill (Plan 077, default-on) |
| `octopus` | — | OCTOPUS octahedral triplet codec — data-oblivious, primary KV compression: -22% to -49% MSE vs SQ, zero calibration (Bench 022, Plan 099, default-on) |
| `turboquant` | — | TurboQuant rotation + uniform codebook — legacy baseline for bench/educate only (Plan 063) |
| `replaid_schedules` | — | RePlaid variance-minimized adaptive schedules — experimental, off by default (Plan 078) |
| `elf_sde` | — | ELF SDE noise injection + logit-normal schedule — 10-22× path diversity (Plan 079, default-on) |
| `cna_steering` | `bandit` | CNA contrastive neuron attribution — sparse circuit discovery + runtime modulation (Plan 087, default-on, GOAT proved) |
| `deep_manifold` | — | Deep Manifold L2/KL residual fixed-point scoring — ResidualRelevanceScorer (Plan 085, default-on, GOAT 6/6) |
| `federation` | `bandit` | Deep Manifold federated KL boundary alignment — KlBoundaryAligner, no data exchange (Plan 085, default-on, GOAT 6/6) |
| `tes_loop` | `bandit` | SimpleTES RPUCG loop — trajectory credit, TrajectoryPruner (Plan 086) |
| `maxsim` | — | MaxSim late-interaction scoring — Σ max_j dot, SIMD-accelerated (Plan 080) |
| `bomber-agent` | `bomber` | Coding agent validator loop (Issue 052) |
| `game_state` | `bomber` | GameState forward model trait + generic MCTS (Plan 056) |
| `bandit_mcts` | `game_state` | Bandit-guided MCTS rollout policy — NFSP/MCTS duality (Plan 067) |
| `percepta` | `ordered-float` | CHT hull cache: upper+lower, HullMeta, tie-break, cumsum |
| `percepta_gates` | `percepta` | + ReGLU, stepglu, multiply, persist primitives |
| `percepta_graph` | `percepta_gates` | + Expression/Dimension DSL, ProgramGraph |
| `percepta_wasm` | `percepta_graph` | + WASM decoder + lowering + interpreter (pure Rust) |
| `percepta_compile` | `percepta_wasm`, `good_lp` | + MILP scheduling + weights + transformer + Futamura |
| `lattice_deduction` | — | LDT Lattice Deduction Transformer — α-intersection pruning, conflict detection, asymmetric elimination (Plan 088, default-on, GOAT 7/7) |
| `delta_routing` | — | Delta Block cross-layer routing — residual block importance routing (Plan 097, default-on, GOAT 6/6) |
| `hybrid_oct_pq` | `planar_quant`, `octopus` | Default KV codec — OCT triplet + PQ 2D Givens rotation (Plan 101, default-on) |
| `planar_quant` | `turboquant` | PlanarQuant 2D Givens rotation KV cache — O(d) vs TQ O(d²) (Plan 100) |
| `iso_quant` | `turboquant` | IsoQuant 4D quaternion rotation KV cache — O(d) vs TQ O(d²) (Plan 100) |
| `dash_attn` | — | DashAttention adaptive sparse hierarchical attention via α-entmax routing (Plan 106, default-on, GOAT 9/9) |
| `mls_aggregate` | — | MLS Multi-Layer Sum aggregation of last K layer residuals (Plan 104, default-on, GOAT 6/6) |
| `gdn2_attention` | — | GDN2 Gated DeltaNet-2 recurrent attention — O(1) decode (Plan 105, default-on, GOAT 14/14) |
| `dreamer` | `bandit` | Auto-Dreamer offline memory consolidation with cadence scheduler + Q-value clustering (Plan 107, default-on, GOAT 8/8) |
| `lt2_looped` | `hla_attention` | LT2 looped inference — weight-shared T-pass loop with hybrid SDPA+AHLA dispatch (Plan 108, default-on, GOAT 8/8) |
| `dmax_spd` | `dllm` | DMax Soft Parallel Decode — hybrid token/mask embeddings with contiguous prefix promotion (Plan 109, default-on, GOAT 7/7) |
| `eqr_convergence` | `elf_sde` | EqR convergence-based rollout selection — Top1Converged picks smallest marginal-change residual (Plan 119, default-on) |
| `subterranean` | `bandit` | Procedure graph compilation — user-defined token-rewriting procedures compiled to zero-cost native code (Plan 110, default-on) |
| `sr2am_configurator` | `bandit`, `g_zero` | SR²AM Configurator Bandit — learned per-turn planning regulation via UCB1 (Plan 112, default-on) |
| `data_gate` | `bandit` | Task-level data gating for self-play training stability (Plan 111, default-on) |
| `tiled_attention` | — | Tiled online-softmax flash attention for CPU SIMD (Plan 115) |
| `parallax_attn` | `tiled_attention`, `newton_schulz`, `katgpt-core/parallax_attn` | Parallax parameterized local linear attention — streaming covariance correction (Plan 135, opt-in) |
| `coda_fusion` | — | CODA fused SIMD kernels — matmul+residual+rmsnorm+activation in single-pass (Plan 103) |
| `moa_inference` | `coda_fusion`, `katgpt-core/moa_inference` | MoA Mixture of Activations — token-adaptive activation mixing over 7-activation dictionary (Plan 158, default-on, GOAT) |
| `stability_metrics` | — | Per-step execution stability instrumentation — P50/P99/CV/stability_score (Plan 102, default-on) |
| `decode_specialize` | — | Stage-specialized decode paths for speculative decoding (Plan 102) |
| `tri_mode` | `dllm` | Tri-Mode inference — AR + Diffusion + Self-Speculation, D2F Drafter Verifier (Plan 089) |
| `unit_distance` | — | Unit Distance GOAT proof — number-theoretic lattice constructions (Plan 090) |
| `plasma_path` | `katgpt-core/plasma_path` | Ternary SIMD matvec — bit-plane ternary weights for SIMD-accelerated matmul (Plan 117, default-on, GOAT 5/5) |
| `parallel_probe` | — | Parallel-Probe 2D — consensus-based parallel branch control for N parallel reasoning branches (Plan 133, default-on, GOAT 7/7) |
| `tf_loop` | `katgpt-core/tf_loop`, `lt2_looped` | Training-Free Loop — pure inference-time mid-stack looping with ODE-motivated damped sub-stepping (Plan 136, default-on, GOAT 4/4) |
| `safe_bandit` | `bandit` | PrudentBanker Safe-Phased Bandit — delay-calibrated safe exploration with bounded regret (Plan 137, opt-in) |
| `stiff_anomaly` | — | Stiff/Soft Subspace Anomaly Gate — eigenvalue decomposition anomaly detection (Plan 138, opt-in) |
| `ega_attn` | — | Energy-Gated Attention — spectral salience gating (Plan 139, opt-in) |
| `cache_prune` | — | CachePrune — SAT + rolling hash + sensitivity masking for KV cache pruning (Plan 140, opt-in) |
| `data_probe` | — | Data Probe Diagnostics — information-theoretic validation with Markov chain analysis (Plan 141, opt-in) |
| `state_source` | `bandit` | State-Source Modelless Distillation — state-visitation tracking + P-UCB selector (Plan 142, opt-in) |
| `skill_opt` | — | SkillOpt — text-space skill optimization framework (Plan 144, opt-in) |
| `proof_cert` | — | Hierarchical GOAT Proof Certificates — formal verification methodology with certificate chains (Plan 145, opt-in) |
| `nexus_elo` | `state_source`, `bandit` | Nexus Elo — Plackett-Luce + P-UCB + goal cache for DDTree/SR²AM (Plan 143, opt-in) |
| `mech_attribution` | `cna_steering`, `ropd_rubric`, `bandit` | Mechanistic Data Attribution — catalyst pattern detection + influence proxy (Plan 111, opt-in) |
| `event_log` | `bandit` | Event-sourced game traces with fork-and-diff (Plan 124, GOAT 22/22) |
| `epiplexity_scoring` | `bandit` | Epiplexity structural information scoring — prequential coding estimator (Plan 130, opt-in) |
| `leo_all_goals` | `katgpt-core/leo_all_goals` | LEO All-Goals Q-value trait framework — `LeoHead`, `AllGoalsUpdate`, `sigmoid_bounded_q` (Plan 155, default-on, SUPER GOAT) |
| `dual_leo` | `leo_all_goals`, `katgpt-core/dual_leo` | Dual LEO teacher-student mixing — `DualLeoMixer` + `AutocurriculumSampler` (Plan 155, default-on, SUPER GOAT) |
| `sigmoid_margin` | `katgpt-core/sigmoid_margin` | Sigmoid margin loss + retrieval margin diagnostic — SigLIP-style softplus, `dim_sufficiency_bound` (Plan 157, Research 123, default-on, GOAT 7/7) |
| `newton_schulz` | — | Newton-Schulz orthogonalization + Muon momentum — 5-iteration cubic fixed-point for optimizer weight matrices (Plan 152, default-on, GOAT 25/25) |
| `river_valley` | — | River-valley diagnostic metrics — subspace ratios, effective rank, update cosine similarity (Plan 152, default-on, GOAT 25/25) |
| `proof_sketch_evolution` | `bandit` | Proof Sketch Evolution — Elo-rated proof population + global goal cache for DDTree/SR²AM (Plan 128, Research 088, opt-in) |
| `datrie_vocab` | — | Double-array trie vocab lookup — zero-alloc trie for ToaST tokenizer (Research 137, opt-in, pending benchmark) |
| `kog_cpu_fusion` | — | Kog AI monokernel CPU fusion — RMSNorm gamma folding + QKV interleaving (Plan 160, Research 139, default-on, GOAT 3/3 Gemma 2 scale) |
| `flashar_anchor` | `dllm` | FlashAR strided anchor-then-fill D2F decoding (Plan 166 T11, opt-in) |
| `flashar_consensus` | `tri_mode`, `plasma_path` | FlashAR consensus tri-mode with ternary thermal paths (Plan 166). **DEMOTED from default-on** (Issue 136, 2026-07-12, removed, see git history): Plan 485 benchmark showed KL 2.9-6.5 (100× worse than Leviathan baseline 0.03). DSpark entropy-skip hybrid dominates on both axes. Opt-in. |
| `budget_adaptation` | — | Compression-adaptive decode budget (Plan 167, default-on) |
| `ilc_distill` | — | ILC iterative latent clustering distillation — synonym-aware DDTree pruning (Research 136 GOAT 6/6, default-on) |
| `hydra_budget` | — | Hydra-aware adaptive layer budget — emergent self-repair layer skipping (Plan 165, default-on) |
| `gepa_reflective` | `bandit` | GEPA-D reflective config evolution — Pareto bandit config evolution (Plan 164, default-on) |
| `phrase_boost` | — | PhraseBoost context trie phrase boosting for DDTree (Plan 164, default-on) |
| `shard_kv` | `spectral_quant`, `turboquant` | ShardKV asymmetric K/V compression — undo RoPE + PCA K path, Hadamard + K-means V path (Plan 147, opt-in) |
| `sleep_consolidation` | `lt2_looped`, `gdn2_attention` | Sleep Consolidation — offline recursive memory consolidation at KV eviction into GDN2 fast weights (Plan 154, default-on, GOAT 14/14) |
| `spectral_hierarchy` | `katgpt-core/spectral_hierarchy` | Spectral hierarchy diagnostic — eigenspace alignment, Haar wavelets, Cauchy interlacing for KG extraction validation (Plan 156, default-on, GOAT) |
| `dual_gram_pca` | `katgpt-core/dual_gram_pca` | Dual-Gram PCA routing for short-sequence calibration (Plan 159, default-on, GOAT) |
| `roofline_cost` | `katgpt-core/roofline_cost` | Roofline cost model for GPU operator runtime prediction — compute/memory/launch bottleneck estimation (Plan 159, default-on, GOAT) |
| `peira_distill` | `katgpt-core/peira_distill`, `bandit` | PEIRA inter-view regressor alignment — collapse-free modelless distillation via EMA covariance (Plan 153 GOAT 7/7, default-on) |
| `parallax_attn` | `tiled_attention`, `newton_schulz`, `katgpt-core/parallax_attn` | Parallax parameterized local linear attention — streaming covariance correction (Plan 135, opt-in) |
| `freq_bandit` | `bandit` | FreqBandit — oscillatory spectral bandit for cyclic pattern detection to adaptive speculative decode (Plan 189, default-on, GOAT 7/7 G189=GAIN) |
| `belief_drafter` | `katgpt-core/belief_drafter`, `papaya` | NextLat Belief-State Speculative Drafter — lightweight 3-layer residual MLP recursive hidden-state prediction for variable-length self-speculative decoding + LatentTransitionCache + BeliefRankPruner (Plan 217, default-on, GOAT 43 tests + 7 benchmarks) |
| `bfcf_lfu_shard` | `bfcf_tree`, `papaya` | BFCF × LFU × Sharding — region-level LFU cache with frequency-aware sharding, batch processing, NeuronShard compound keys, emotion-aware eviction, KG triple transitions (Plan 218, default-on, GOAT 44 tests + 10 benchmarks) |
| `caddtree_budget` | `spec_cost_model` | CaDDTree — Cost-Aware Adaptive DDTree Budget Selection (Plan 219, 7 GOAT tests, default-on) |
| `hardware_aware_scheduler` | — | Hardware-Aware Prefix Scheduler — multi-request verification budget allocator (Plan 339, Issue 003, DSpark §3.2.2). Global sort + greedy admission + non-anticipating early-stop (Appendix A correctness theorem). Opt-in until a real multi-request batch caller exercises the synthetic GOAT gate. |
| `manifold_pruner` | — | ManifoldPruner — ManifoldE point-to-manifold soft validity scoring + kernel-tricked relevance for ScreeningPruner (Plan 234, opt-in, GOAT G1 FAIL) |
| `sense_composition` | `katgpt-core/sense_composition` | KG Latent Octree NPC sense modules — ternary bit-plane projection, GM override, hot-swap, bandit feedback (Plan 221, opt-in) |
| `shard_embedding` | — | 🪦 **DEPRECATED (Issue 139)** — JL random orthogonal projection [f32;64]→[f32;8] for O(1) cosine similarity shard lookup (Plan 230). Violates JL lower bound 200× at m=8; marked `#[deprecated]`, zero runtime consumers. |
| `slod` | `katgpt-core/slod`, `spectral_hierarchy` | SLoD Spectral Level-of-Detail Pruner — Poincaré ball hyperbolic geometry + heat diffusion tier routing (Plan 235, default-on, GOAT G1–G6) |
| `schema_centroid` | `katgpt-core/schema_centroid`, `dep:papaya` | Schema Centroid — per-class embedding centroids for informed KG entity init (Plan 237, default-on, GOAT 7/7) |
| `bake_precision` | `katgpt-core/bake_precision`, `dep:papaya`, `sense_composition` | BAKE Precision-Gated Bayesian Embedding — per-dimension precision tracking, O(8) arithmetic (Plan 236, opt-in, GOAT 10/10 but marginal) |
| `nf_flow_score` | — | NFCoT FlowScore — modelless normalizing flow density scoring for speculative candidates (Plan 229, opt-in) |
| `nf_flow_gate` | `nf_flow_score` | NFCoT adaptive EMA acceptance criterion (Plan 229 T3, opt-in) |
| `nf_flow_budget` | `nf_flow_score` | NFCoT sigmoid-weighted speculative depth allocation (Plan 229 T4, opt-in) |
| `nf_flow` | `nf_flow_score`, `nf_flow_gate`, `nf_flow_budget` | NFCoT parent feature — enables score + gate + budget (Plan 229, opt-in) |
| `union_bound_confidence` | — | Union Bound Confidence — additive branch confidence via Boole's inequality (Plan 231, default-on, GOAT 6/6) |
| `pathway_tracker` | — | PathwayTracker — intrinsic pathway stability detection (Plan 231, default-on, GOAT 7/7) |
| `federation_composer` | — | FederationComposer — explicit pruning with residual early termination (Plan 231, default-on, GOAT 7/7) |
| `collapse_aware_thinking` | `selectivity_router`, `thinking_cot`, `bandit` | Collapse-aware adaptive thinking — runtime reasoning collapse detection + early exit (Plan 212, default-on) |
| `cgsp` | `bandit`, `collapse_aware_thinking`, `data_gate`, `breakeven_routing` | Curiosity-Guided Self-Play — modelless Solver/Conjecturer/Guide triad with collapse recovery + BLAKE3-committed personality snapshots (Plan 274, Research 240 — **opt-in**: GOAT gate run, G2/G3/G4/P2/P3/G6 pass; G1 is informational because CGSP is curiosity-driven not target-seeking — see `.benchmarks/274_cgsp_goat.md`) |
| `substrate_gate` | `katgpt-core/substrate_gate` | SubstrateGate — inference-time routing via substrate conditions (Plan 216, default-on) |
| `llmexec_guard` | — | Entropy-driven verification budgeting (default-on) |
| `outlier_guard` | — | Model-load-time outlier injection detection via KS D-statistic (default-on) |
| `segment_checkpoint` | — | Segment-level checkpoint/rollback for speculative decoding (default-on) |
| `trust_region_spec` | — | Trust-region speculative verification (default-on) |
| `precision_aware_draft` | — | Precision-aware draft selection (default-on) |
| `self_distilling_bandit` | — | Self-distilling bandit arms (default-on) |
| `static_cal_tables` | — | Pre-computed calibration tables for quantization (default-on) |
| `targeted_precision` | — | Targeted precision allocation for KV cache (default-on) |
| `egcs` | — | Expert-gated channel selection (default-on) |
| `nds_proxy` | — | NDS Proxy — normalized difference score proxy for routing (Plan 186, default-on) |
| `rat_plus_bridge` | `katgpt-core/rat_plus_bridge` | RAT+ Recurrence Bridge via GDN2 state for modelless dilated inference (Plan 225, opt-in) |
| `swir_switch_thinking` | `thinking_cot` | SwiR Switch-Thinking — explicit↔latent reasoning mode controller driven by entropy trends, asymmetric dwell windows + switch-count overthinking guard (Plan 275, Research 241, **DEFAULT-ON** since Plan 313 T6.2, 2026-06-27): G2 token-efficiency 1.32×/1.37×/1.43× at n=3/5/10 on Gemma 2 2B + MATH-500 (gate ≥1.3×, all pass) with tuned config (w_e_to_l=32, c_max=64). G1 accuracy blocked by model capability (Gemma 2 2B too small, not a SwiR design flaw); G3–G6 all pass (3.1ns/step, convex-hull 1000/1000, no-regression, kurtosis escape). Token efficiency is the primary value prop. |
| `micro_belief` | `katgpt-core/micro_belief` | MicroRecurrentBeliefState — per-entity recurrent state trait + attractor/leaky/latent-thought kernels + BLAKE3 snapshot + bridge (Plan 276, Research 242, **opt-in**: G1.1/G1.2/G1.3/G1.5 pass; G1.4 latency FAIL ~273ns; G2.1 coherence FAIL — attractor demoted to Gain, trait unification + LeakyIntegrator are the promotable outputs) |
| `sink_aware_attn` | `data_probe` | Sink-Aware Attention — dual-policy sigmoid gate (Plan 287 Phase 3, Research 258, arxiv 2606.08105). Implies data_probe for the classifier primitive. **Opt-in**: default stays Uniform pending G2/G3 GOAT gate. Different paper + mechanism than `depth_invariance` (target-side sink classification vs drafter-side magnitude accumulation). |
| `depth_invariance` | `katgpt-core/depth_invariance` | Depth-Invariance Diagnostic + MagnitudeRegularizedResidual — root-cause counterpart to BeliefRankPruner / GainCostLoopHalter / latent_functor/reestimation / micro_belief/coherence_bench (Plan 306, Research 286, arxiv 2605.09992). Detects DepthSpecificRefinement / Collapsed / DepthInvariant on flattened `&[f32]` state chains. **DEFAULT-ON** (Plan 306 T7.4, 2026-06-23): G1 (8 correctness tests) + G2 (paper finding reproduced on random-init BeliefDrafter) + G3 (negative control on AttractorKernel + positive control on unclamped leaky) + G4 (re-spec to absolute-latency at HLA scale, all PASS) + SIMD inner-loop landed. Zero runtime cost unless invoked. |
| `temporal_deriv` | `katgpt-core/temporal_deriv` | Temporal Derivative Kernel — dual fast/slow EMA surprise signal driving 4 consumers (HLA companion, δ-Mem write gate, collapse detector, derivative curiosity) via a unified α-pair (Plan 277, Research 243, arXiv:2606.08720, **default-on**, GOAT 4/4) |
| `hippocampal_cache` | `katgpt-core/hippocampal_cache` | HOLA Hippocampal Exact KV Cache — surprise-evicted (β·‖e‖) bounded KV cache with decoupled RMSNorm-γ softmax read, complementing the GDN2 recurrent state (Plan 395, Research 378, arxiv 2607.02303, **opt-in**: G1–G4 modelless PASS, G5 perplexity deferred to riir-train Issue 038) |
| `hga` | `katgpt-core/hga` | Hierarchical Global Attention — chunk→group→token routing with mixed-RoPE summaries + tiered route-and-fetch KV store (Plan 397, Research 379, arxiv 2606.30709, **opt-in**: G1/G3/G5 PASS, G2-proxy FAIL on random-key NIAH — same class as MSA R225 GOAT-FAILED; full G2 transformer loss-gap deferred to riir-train). `TieredKvStore` trait ships always-on as generic route-and-fetch primitive. |
| `faithfulness_probe` | — | FaithfulnessProbe — causal intervention diagnostic for injected memory (Plan 278, Research 244, opt-in) |
| `triggered_injection` | — | TriggeredInjectionGate — sigmoid-thresholded inject/skip hot-path gate (Plan 278, Research 244, default-on, G3 PASS — saves compute, matches quality) |
| `manifold_power_iter_router` | — | Manifold Power Iteration MoE Router — one-shot router-row conditioning at snapshot swap via shared `spectral_retract` helper (Plan 279, Research 246, arxiv 2606.12397, **DEFAULT-ON** since Plan 279 Phase 4 GOAT 9/9 green, G1 λ-alignment + G2 MaxVio reduction + G3 zero per-token overhead) |
| `quantile_balance_router` | — | Quantile Balancing MoE Router — one-shot per-expert bias β at snapshot swap via alternating-coordinate descent on the balanced-assignment LP (Plan 455, Research 447, Su blog Feb 2026 + Marin 32B validation, **DEFAULT-ON** since Plan 455 Phase 3 2026-07-17): G1–G8 12/12 GOAT gate green + Phase 3 head-to-head Case C (composed pipeline `R'=MPI(R) → β=QB(X·R'^T) → top-k(s−β)` strictly Pareto-dominates either alone: λ 0.65→0.99 from MPI, MaxVio 1.84→0.00 from QB on orthogonal axes) |
| `cs_kv_probe` | — | CS-KV-Importance Probe + Density-Budget Interpolator — compressed-sensing KV-group importance via ablation + Lasso, sigmoid-gated top-K application (Plan 280, Research 247, arxiv 2606.13594, **opt-in**: G1 CS-beats-random + G2 sparse-vs-dense duality shape + G3 K(ca) monotone/bounded) |
| `self_advantage_gate` | — | Self-Advantage Recursion Gate — dead-compute detector via pre/post-recursion log-ratio (Plan 283, Research 250, arxiv:2511.16886, default-on, GOAT 4/4 PASS) |
| `funcattn` | `tiled_attention`, `katgpt-core/funcattn` | Functional Attention — closed-form Tikhonov k×k spectral transport operator (Plan 286, Research 257, arxiv 2605.31559, **DEFAULT-ON** since 2026-07-07): dual form `(1-α)·K̃ᵀK̃ + α·I_d` convex-combo regularization, sigmoid-basis default per AGENTS.md. Promoted after G6 LLM-domain gate PASS — the prior FAIL was an Issue 049 test-data-gen artifact (admitted ~12.5% degenerate `a==b` constant sequences at V=8, corrupting the learned basis into a spurious 0.969 plateau). D4 fix rejects degenerates; post-fix FUNCATTN=1.000 SDPA=1.000 @ release 600 FD-SGD steps (debug 40-step: FUNCATTN 0.9297 > SDPA 0.6719). 6/6 GOAT gates green (G1 correctness, G2 perf 10.9× sample-efficiency, G3 no-regression, G4 alloc-free, G5 feature-isolation, G6 LM-domain parity). Modelless (no training). Heavy downstream use: riir-ai plans 318/329/330/309/310/321 consume `C` as transport substrate; **riir-ai Issue 533 (2026-07-17)** consumes `funcattn_forward` + the `k` parameter as the attention-rank LOD row of the Thermal LOD coordinator (tier → k, data-derived from Plan 332 k-sweep elbow). |
| `functional_substitution_gate` | `katgpt-core/functional_substitution_gate`, `funcattn`, `faithfulness_probe` | Head Substitution Gate — IoU cheap-proxy + cached FaithfulnessProfile veto decision wrapper around FuncAttn (Plan 353, Research 353, arxiv 2606.19317). **Opt-in Gain-tier**: G1+G3+G4 green, G2 synthetic green (Spearman ρ ≤ −0.9 across seeds/sizes), G2 real-head deferred to riir-ai. Gate wrapper around FuncAttn; not a new primitive (the original `ProgramSynthesizedHead` draft was dropped after re-review identified FuncAttn as the existing primitive surface). |
| `chain_fold` | `thinking_cot` | ThoughtFold chain folding — inference-time CoT step pruning (Plan 195 GOAT 16/16, Plan 228 78% reduction) |
| `clr` | — | CLR Claim-Level Reliability runtime — `(mean_m v_k,m)^M` nonlinear reliability vote over claim embeddings + Long2Short brevity tiebreak + learning potential + MGPO sampling weight (Plan 284, Research 255, arxiv 2606.16140, **default-on**, GOAT G1–G5: CLR beats majority +78pp, ECE 0.0087, K=32 vote 4–5µs, zero-alloc vote internals, feature-isolated) |
| `ict_branching` | `katgpt-core/ict_branching` | ICT Distributional Branching-Point Detector — `collision_purity(π) = Σ π²` (proven unconditionally monotone, ICT §A.2.5 — H₁ wrong below π > e⁻¹≈0.37) + Jensen-Shannon divergence to group mean + `BranchingDetector` top-k% selector (Plan 294, Research 270, arxiv 2606.19771, **opt-in until G3+G8 pass**): G1 PASS paper Fig 1a bifurcation; G2 BORDERLINE-FAIL median 37.5% (paper's 10% is LLM-token-specific — sweep `k_percent` per-domain); G3 ⭐ PASS Spearman ρ(H₁, JS-uniqueness) = 0.0652 95% CI [-0.017, 0.150] < 0.5 — JS structurally orthogonal to H₁ (Super-GOAT proceeds); G4 PASS 1.96µs/call (target ≤50µs); G5 PASS 0 allocs; G6 PASS feature-isolated; G10 PASS Bebop H₁→H₂ acceptance-forecast upgrade MAE 0.402<0.423 on long-tail. Stays opt-in pending G8 (riir-ai Plan 324 runtime fusion). |
| `induced_cwm` | `katgpt-core/induced_cwm` | Induced Code World Model kernel primitive — `InducedCwmKernel: GameState` marker + `CwmCommitment` (BLAKE3) + `BeliefInferenceFn<S>` + `TransitionUnitTest` (Plan 296, Research 275, arxiv 2510.04542, **opt-in**: G1–G4 GOAT 4/4 PASS, ready for downstream consumption; LLM-induction pipeline is private — riir-ai Plan 326) |
| `induced_cwm_ismcts` | `katgpt-core/induced_cwm_ismcts`, `induced_cwm` | Information-Set MCTS over an induced CWM + belief fn (Plan 296 Phase 2) |
| `induced_cwm_tournament` | `katgpt-core/induced_cwm_tournament`, `induced_cwm` | Value Function Tournament — round-robin arena-play selector over `StateHeuristic` candidates (Plan 296 Phase 3) |
| `subspace_phase_gate` | `katgpt-core/subspace_phase_gate` | Participation ratio + numerical rank + N≥d phase-transition gate + runtime Jacobian SVD (Plan 301, Research 279, arxiv 2409.02426). Pure numeric substrate; consumed by Plan 312. **DEFAULT-ON** (Plan 301 Phase 5 T5.1, 2026-07-02) — G1 PASS + Phases 3–5 complete; also pulled transitively via `viable_manifold_graph` (DEFAULT-ON). |
| `alien_sampler` | `katgpt-core/alien_sampler` | Coherence × Availability frontier ranking — `AlienSampler<V,C,A>` z-scored fusion + `MedianTopMAvailability` median-of-top-m cosine rule (Plan 311, Research 293, arxiv 2603.01092). **🪦 GOAT FAILED 2/4** — G1+G2 fail (β phase-transition at β≈0.4); G3 PASS post-rayon (4.56×); G4 PASS. Opt-in for paper reproduction. |
| `viable_manifold_graph` | `katgpt-core/viable_manifold_graph`, `subspace_phase_gate` | Discrete safe-manifold navigation — `pullback_volume` + `SafeManifoldGraph` (CSR adjacency) + `manifold_geodesic` / `manifold_random_walk` / `manifold_curiosity_walk` (Plan 312, Research 294, arxiv 2206.00106). **DEFAULT-ON** — G1–G7 correctness all PASS + perf bench PASS post-CSR (random walk 7.10 ns/step, 14× under 100 ns target). |
| `ac_prefix` | `katgpt-core/ac_prefix` | AC-GPT arbitrary-conditional prefix — mask builder + sequence augmenter turning any causal Transformer forward into single-pass `p(xe | xc)` (Plan 313, Research 295, arxiv 2606.14943). Three-region attention rule, branch-free `attends(i,j)`, bit-packed `AcPrefixMask`. **DEFAULT-ON** — G1 (buffer construction bit-identical) + G2 (27.46× speedup vs iterative-MLM) + G3 (empty-prefix no-regression) + G4 (alloc-free hot path) all PASS. G1 reformulated: original "matches iterative-MLM to 1e-4" is a trained-model property (holds post-LoRA, riir-train's job); modelless G1 tests buffer construction. |
| `closed_unit_compaction` | `katgpt-core/closed_unit_compaction` | Closed-Unit Compaction Gate (CUCG) — generic rubric-gated trajectory compaction primitive (Plan 333, Research 300, arxiv 2606.23525 SelfCompact). Fires summarization at structurally-safe moments (closed-unit ∧ summarizable ∧ progress ∧ ¬stuck) via sigmoid projections + `FireRule` Boolean tree + `Backstop` token-pct + optional `skip_if_reliable` CLR fuse, instead of fixed token thresholds. **DEFAULT-ON** — 7/7 GOAT gates PASS: G1 recall=1.000/FDR=0.000, G2 50% suppression, G3 probe ratio=1.00, G4 zero-alloc, G5 feature isolation, G6 0 softmax, G7 `can_freeze` isomorphism (Super-GOAT: trajectory compaction and shard freeze are the same primitive, proven structurally). `evaluate()` 8.91ns <50ns, 112.9M/s ≥50M. Zero runtime cost unless a caller invokes evaluate. |
| `committed_field_blend` | `katgpt-core/committed_field_blend` | CommittedFieldBlend — sampling-invariant per-entity MoE: frozen sigmoid blend of N archetype operator fields, weights computed ONCE from a trajectory summary + BLAKE3-committed (Plan 321, Research 302, arXiv:2510.00621 FAME). Defining property: **sampling invariance** (FAME Prop. 3) — dense vs sparse observation of same trajectory → identical committed `pi` + identical dynamics. Implies `personality_composition` (reuses sigmoid + `simd::simd_fused_scale_acc`). Closed-form Lipschitz safety bound (`max_k sigmoid(pi_k/tau)·L_k`, FAME Lemma 1). **DEFAULT-ON** (2026-06-28) — G1–G5 GOAT gate ALL PASS (G2 sampling invariance holds across 100 entities, worst-case Δpi=1.19e-6; G4 zero-alloc on apply + commit; G5 BLAKE3 reproducible + tamper-detecting). Runtime validation also shipped: riir-ai Plan 336 G6a–G6e + G7a ALL PASS (2026-06-26). Modelless gain (closed-form sigmoid projection + BLAKE3 commit, no training). Zero runtime cost unless a caller invokes commit/apply_blended. |
| `qgf` | `qgf_oracle`, `qgf_projector` | QGF Test-Time Q-Guided Flow parent — test-time Q-gradient guidance (Plan 268, Research 236, arxiv 2606.11087). Parent feature; enables the trait + projector only. The drafter (F1) and adaptive (F4) are separate sub-features. |
| `qgf_oracle` | — | `QGradientOracle` trait — drop-Jacobian critic gradient `∇_a Q(s, â_1)` (Plan 268 F3). Four shipped impls: `NoGuidanceOracle` (freeze-tier no-op, zero confidence), `LeoHeadOracle<H>` (Plan 268, single LEO teacher head — `leo_all_goals`), `FlowFieldOracle` (Plan 268, owned `FlowField` `(dx,dy)` lookup — `flow_field_nav`), and **`DualLeoOracle<H1,H2,M>`** (Plan 467 / Proposal 007, LEO teacher + UVFA student α-mix at the gradient level — `leo_all_goals + dual_leo`; encodes the Plan 460 "no operator between mix and consumer" invariant by construction). **Plan 467 G1–G4 PASS mechanistically; G5 measured FAIL on synthetic data** (riir-ai Bench 553, 2026-07-18: dual 0.00% vs single 0.50% on T7 Go puzzles, but the correctness invariant `b ≡ a` held bit-identically — mechanism correct, quality gate FAILs because synthetic data produces near-flat Q-fields). **G5 measured FAIL on civ real networks too** (riir-ai Bench 558, 2026-07-19: dual +2.69% vs single 35.68% → 36.64% on civ action-prediction, ≥3% gate — fourth-axis stop rule; the civ dual-LEO investigation is fully closed per Research 322 — the alternative-critic escape hatch was category-confused). Stays opt-in with documented unproven G5 across both synthetic and civ real-network regimes; reopens only on seal integration gain, new game domain positive G5, or Q-vs-forecast research breakthrough. |
| `qgf_projector` | `speculative_generator` | `FirstOrderProjector` — one-step Euler projection `â_1` of the final output (Plan 268 F2). |
| `qgf_drafter` | `qgf` | `QGuidedDrafter` — `tilt_logits` hot path (`logits += w · ∇Q`, SIMD AXPY, zero-alloc) + tier-routing policy `route_for` (Plan 268 F1 + Phase 4 T8). **Opt-in**: katgpt-core mechanism gates G1–G5 PASS (G1 non-circular via 2 negative controls, G4 0 allocs/2000 calls, G5 sigmoid bounded/finite/monotone); downstream selling-point gates (Sudoku/DDTree/Bomber) deferred to riir-ai. |
| `qgf_adaptive` | `qgf_drafter` | `VarianceAdaptiveGuidance` — per-query sigmoid `1/β = sigmoid(k·(confidence − threshold))` (Plan 268 F4). Novel extension beyond the paper's fixed `1/β`. **Opt-in** until real-world validation on Bomber arena. |
| `cross_stage_relocation` | `causal_head_importance`, `katgpt-core/cross_stage_relocation` | Cross-Stage Residual Relocation Operator + Permeation-Map Diagnostic — Knowing-Using Gap (Plan 431, Research 417, arxiv 2607.08393). Two modelless primitives: (1) `permeation_scan_into` 2D `(src,dst)` intervention heatmap reusing Plan 358's `direct_effect_importance` + two-cluster classification; (2) `RelocateOp` applied operator with paper's fixed `(0.82L→0.45L)+(0.10L→0.45L)` default (`RelocatePair::LateEarly`, 58–75% oracle recovery). **Opt-in** — G1–G6 PASS for katgpt-rs scope (scan 10–25% faster than hand-rolled; operator <0.03% of forward pass; 0 allocs); G7 (58–75% recovery transfer to our substrate) deferred to Phase 3 PoC in `riir-poc/`. Stack slot: intervention/diagnostic. |
| `occupancy_ratio` | — | FORE Occupancy-Ratio Estimator — adjoint-Bellman KL-contraction occupancy-ratio probe (Plan 438, arxiv 2607.05375). **Phase 5 CLOSED (2026-07-15)**: all tasks complete; FORE stays opt-in (Baird-MRP G1 PASS but no downstream consumer wired). Ships `OccupancyRatioEstimator` + KL-projection fit loop (Algorithm 1) + recos MAG `TransferMetric` cold-path diagnostic (Plan 437 Phase 3/4 DONE). |
| `ane_fused_chain` | — | ANE Fused-Chain Cost Model — dependency-aware overlap estimator for chained ANE operations accounting for DMA/Port/Kernel overlap (Plan 439, arxiv 2606.22283). Real M3 Max validation landed (Phase 2.5); consumer integration (riir-engine) shipped (Phase 4). **DEFAULT-ON** (Plan 439 Phase 2 GOAT G1–G5 PASS, promoted 2026-07-14). |
| `multi_agent_path` | — | Lifelong LaCAM Local Guidance Substrate — modelless, training-free, receding-horizon windowed multi-agent pathfinder (Plan 440, Research 424, arXiv:2605.16855 Arita & Okumura AAAI 2026). `LifelongLaCam<P, C, G>` = PIBT one-step generator + per-agent space-time A* guidance (BFS distance field) + warm-start schemes (LLLG_Π / LLLG_Φ / LLLG_∅) + one-step blocking-count hindrance estimator. **Five pluggable seams** (`CostFn`, `LocalGuidanceSource`, `WarmStartScheme`, `HindranceEstimator`, `FlowField`) for the Super-GOAT fusion (riir-ai/318: HLA × Crowd MCGS × P350). Issue 148 upgraded all 4 benchmark maps to real MovingAI files (ht_chantry improved 0.09→0.27); Issue 149 added the `FlowField<P>` seam for 1-wide corridor direction assignment (correct + tested, near-zero effect on real maps — game corridors are 2-wide). Pure heuristic — no training, no backprop. **Opt-in** — G3 (no-regression, 1601 tests) + G4 (latency, 467ms median @ 1000 agents on real MovingAI maps) PASS; G1 PARTIAL (2/4 maps — warehouse/ht_chantry fail, greedy PIBT lacks priority inheritance); G2 FAIL (warm-start not consumable by greedy rollout). Promotion deferred until G1/G2 unblock and riir-ai/489 G5–G7 fusion gates validate the Super-GOAT claim. |
| `flow_field_nav` | `katgpt-core/flow_field_nav` | Fourier-Smoothed Flow Fields for LEO crowd navigation (Plan 242, GOAT PASS 46.9%). `FlowFieldCache::get_or_compute<H: LeoHead>` builds a smoothed navigation field from raw `Q_LEO[:,:,g]` slices via `LeoPotentialGrid::from_q_values` + FFT low-pass + finite-difference gradient + unit-length normalization. **Opt-in** — GOAT 46.9% quality gain on synthetic 2D landscape. **Dual-LEO fusion paths** (composition of `dual_leo` + `flow_field_nav`, both already default-on): Plan 459 ships `get_or_compute_dual` (pre-max Q-slice α-mix — G1–G4 PASS, **G5 FAIL** honestly, 25.9% stuck reduction at α=0.1 short of 30% gate; demoted to compatibility); Plan 460 ships `get_or_compute_dual_postmax` (post-max potential α-mix — **G5' PASS at α=0.10: 31.5% stuck reduction; PROMOTED as the recommended dual path**). "Promoted" = doc-level recommendation, NOT a feature-gate promotion (`flow_field_nav` and `dual_leo` were already default-on; the dual path is a sibling API, not a replacement). Real-network quality gain requires riir-games-civ wiring (CivLeoNet + UVFA wrapper). See [`.benchmarks/459_flow_field_dual_leo_mixer_goat.md`](../../.benchmarks/459_flow_field_dual_leo_mixer_goat.md) + [`.benchmarks/460_flow_field_dual_leo_postmax_goat.md`](../../.benchmarks/460_flow_field_dual_leo_postmax_goat.md). |
| `simd_lut_dequant` | — | Software SIMD LUT-accelerated dequant distilled from StreamDQ's hardware DQB (Plan 452, Research 418 §2.3, arxiv 2607.11262). **Split GOAT decision:** the fused `dequant_dot_via_lut` kernel wins **4.58×** over the two-step path (NEON FMA + no buffer spill) → **DEFAULT-ON**; the plain `dequant_via_lut` is **3.5× slower** than the arithmetic cast on NEON (scalar gather, no native instruction) → opt-in infrastructure for future FP8/INT8. G1 bit-exact, G3 no-regression, G4 0 allocs (stack `[f32; N]` LUT + caller-owned out). Cross-repo: `simd_lut_q4k` default-on in riir-engine (Plan 486 T3.3). |
| `lacam_escalation` | `multi_agent_path` | Bounded one-step LaCAM — the constraint-tree search from Okumura 2023 applied to a single tick, replacing the fake "LaCAM escalation" (shuffled-priority retries). The critical insight (Research 441, from reading `Kei18/lacam`): LaCAM = recursive PIBT **+ constraint tree** — only the recursive PIBT half was tried before (Issues 140/143 collapsed throughput); the constraint tree bounds the recursion. **Opt-in** — G6c collision-freedom 1.000 (37.5%→100%), G-col vertex rate 0.0% (Issue 154 fixed), G-PI no-collapse 0.69, G3/G4 PASS; **G1 3/4 maps** (ht_chantry 0.28 marginal — one-step resolves single-tick collisions but not multi-step maze detours). **Issue 546 (2026-07-18) added the multi-step path** as `EscalationBudget::multistep_default()` (stuck-agent targeting + depth 8 + 100ms/100K-node budget) — opt-in escalation; default behavior remains Plan 453 one-step (bit-identical). Measured +0.6% throughput / +2.5% latency on ht_chantry-real: marginal, **G1 not closed** (corridor-queue deadlocks are structurally too long for any bounded-depth search). Pair with Proposal 006 (bi-directional flow field) for the full G1 close. |
| `interpolation_geometry` | `katgpt-core/interpolation_geometry` | Interpolation Geometry — iMAUVE + 5-way intervention probe for committed latent substrates (Plan-158 / Research 445, Prabhudesai & Geng, *Latent Thought Flows with Text Compression*, Jun 2026). Generic `LatentSpace` trait abstracting over HLA `[f32;8]` / `style_weights[64]` / archetype-blend π / KarcShard / ZoneGeometryPod / MerkleFrozenEnvelope (the six substrates cataloged in Research 445 §2.6). Two protocols: `imauve_score` (nearest-neighbor midpoint coherence — the paper's headline metric, Pearson r=0.99 with downstream quality) + `intervention_battery` (matched/shuffled/zero/mean/noise probe extending Plan 278's FaithfulnessProbe to per-entity committed state). **Opt-in** — three-pressure audit (Q1 summarize-vs-route / Q2 runtime-depends-on-latent / Q3 local-context-vs-bypass) PASS for all six substrates; see [Benchmark 456](../../.benchmarks/456_interpolation_geometry_goat.md) + [`.docs/04_calibration/interpolation_geometry.md`](../04_calibration/interpolation_geometry.md). Pure modelless evaluation methodology — NOT a training primitive. |
| `grapem_rodrigues` | `katgpt-core/grapem_rodrigues` | GRAPE-M Rank-2 Rodrigues Exponential — O(d) closed-form application of `exp(n·ω·L)` for arbitrary rank-2 skew generator `L = abᵀ − baᵀ` (Research 446 / arXiv:2512.07805 §2.3). Pure modelless float arithmetic on a user-supplied plane `(a, b)`. Subsumes `phase_rotation`'s scalar-broadcast 2D rotation as the canonical-basis special case. **Opt-in** — G1 bit-identical to materialized `expm(L)`; G2 latency `< 2× phase_rotation_gate_into`; G4 zero-alloc. See [Benchmark 457](../../.benchmarks/457_grapem_rodrigues_goat.md). Promotion deferred pending a hot-path consumer. |
| `position_group_action` | `katgpt-core/position_group_action`, `grapem_rodrigues` | Unified `PositionGroupAction` trait — RoPE / ALiBi / FoX / Wall / NoPE / GRAPE-M as instances of `G(n) = exp(n·ω·L)` (Research 446 / arXiv:2512.07805 §2.2 + §4.1). Vocabulary bridge for position-encoding-agnostic tooling (KV compaction, attention matching). Hot-path code keeps using `PositionFreeCompactor` / `WallDiagonalGate` directly; the trait is for cold-path interop. **Opt-in** — G3 no-regression (existing RoPE/Wall paths unchanged when feature off). See [Benchmark 458](../../.benchmarks/458_position_group_action_goat.md). |
| `grape_ap_vector` | `katgpt-core/grape_ap_vector` | GRAPE-AP Vector-Similarity Path-Integral Decay Gates — content-aware extension of Wall Attention's scalar prefix-sum gates (Research 446 / arXiv:2512.07805 §5). `ψ_h(t,ℓ) = α·g(⟨p_t, R_ℓ·p_ℓ⟩/d)` with `g=log_sigmoid` — tokens whose positional embedding matches the query's decay slower. Maintains per-head prefix sum along causal path. Wall is the scalar special case (endpoint-independent embeddings). **Opt-in** — G2 latency overhead `< 1.5×` Wall's scalar path; G4 alloc-free after scratch init. See [Benchmark 459](../../.benchmarks/459_grape_ap_vector_goat.md). |
| `grape_joint_lift` | `katgpt-core/grape_joint_lift`, `grapem_rodrigues` | GRAPE Joint Lift — `GL(d+2)` block-diagonal composition of rotary (GRAPE-M) + additive (GRAPE-A, paper §4.1) into a single group action per Appendix E of arXiv:2512.07805. One-pass `score_into`: `q^T·exp(m·ω_rot·L)·k/√d + m·ω_add·(softplus(v·q/√d) + softplus(u·k/√d))`. Closes the GRAPE composition story: today Wall *replaces* RoPE; this primitive proves they *compose* into a single one-parameter subgroup of `GL(d+2)` while preserving the exact relative law. Decoupled `omega_rot`/`omega_add` is a strict generalization of the paper's shared `ω`. **Opt-in** — G1 bit-identical to manual composition + relativity; G2 latency smoke; G4 alloc-free after `new`. See [Benchmark 460](../../.benchmarks/460_grape_joint_lift_goat.md). |
| `causal_identification` | `katgpt-core/causal_identification` | Causal-ID — Algorithmic Syntactic Causal Identification (Plan 457, Research 450, arXiv:2403.09580 Cakiqi & Little 2024). Pure modelless graph rewriting on ADMGs with bidirected confounders: `identify(Y, do(A))` returns the interventional signature backbone `Y⋆ = An(Y in G[V\A])` via the recursive Shpitser-Pearl ID algorithm. Four submodules: `types` (`NodeId` BLAKE3 `[u8;32]`, `Admg`, `AdmgSignature` inline-ArrayVec<32>→heap fallback, `IdentificationError` with hedge pair), `fixing` (`districts`, `ancestors`, `fix_node`, `try_fixseq` greedy fixing sequence), `identify` (the 6-step recursive ID), `subgraph` (bounded-BFS to keep the `O(k²)`–`O(k³)` algorithm on a ≤32-node subgraph). **DEFAULT-ON** (Plan 457 Phase 5 promotion, 2026-07-18): Phase 2 GOAT gate G1+G2+G3+G4 ALL PASS — G4 closed by Issue 183 (Scratch refactor cut per-call allocs 284→198, -30%, latency 8.26→6.07µs, -27%) + Issue 184 / Benchmark 466 (`districts()` + `try_fixseq` graph-construction allocators eliminated via callback-based `for_each_district_with_buffers` + workspace-based `try_fixseq_into`; allocs further reduced 198→133/call, -33% more, -53% cumulative from 284). Phase 4 T4.5 synthetic Consumer A bench cleared T4.7 promotion gate (71.7% non-trivial Ok rate, 43/60 queries on a 100-node game-world KG with 3 faction confounder cliques). Promotion follows the codebase pattern (manifold_bandit P370, set_attention P354, poincare_navigator P449). Offline-only (24µs on 13 nodes is well outside the 20Hz tick). Pure modelless — no training, no learned params. `blake3` + `arrayvec` already non-optional. Zero runtime cost unless invoked. See [`.benchmarks/465_causal_id_alloc_free_scratch.md`](../../.benchmarks/465_causal_id_alloc_free_scratch.md) + [`.benchmarks/466_causal_id_p4_zero_alloc.md`](../../.benchmarks/466_causal_id_p4_zero_alloc.md) + [`.benchmarks/464_causal_id_consumer_a_synthetic.md`](../../.benchmarks/464_causal_id_consumer_a_synthetic.md). |
| `conformal_predictive_intervals` | `katgpt-core/conformal_predictive_intervals` | Conformal Predictive Intervals — modelless UQ overlay wrapping any `PointForecaster` with a per-channel × per-horizon-bucket exp-recency-weighted residual ring buffer, reading empirical quantiles to produce coverage-guaranteed predictive intervals `[point+q_{α/2}, point+q_{1−α/2}]` (Plan 340, Research 322, arxiv 2605.03789 CSP + 2606.09473 "Report the Floor"). CRPS / Winkler / empirical-coverage metrics (`conformal::metrics`). The `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` with m=1 is the **canonical conformal-naive floor** — every UQ-bearing primitive's GOAT gate MUST beat it (Issue 010 "Report the Floor" rule, codified in AGENTS.md Feature Flag Discipline). No training, no learned params. **DEFAULT-ON** (Plan 468 promotion, 2026-07-20): primitive-level G1–G4 GOAT PASSed (Bench 340, 2026-06-30) — coverage [0.9445, 0.9493] ∈ [0.93, 0.97], `interval_into` H=1 **642 ns** (≤ 1 µs target), 0 allocs/100 calls, bit-reproducible; runtime-consumer promotion gate satisfied by Bench 564 (MCTS collapse G3 PASS — per-NPC calibrated τ beats fixed magic number on F1) + Bench 565 (Salience Tri-Gate G3 PASS — interval-width Delegate nudge dF1=+0.3145 at 6.3× gate margin, dFP=−0.8155; Plan 513 width-definition fix vindicated bit-identically). Two consumers FAILED (Bench 562 curiosity — wider than 5×EMA; Bench 563 sleep-time — distribution-level summary loses cycle info); Cargo.toml language required only one PASS, two landed. **Consumer-level gates STAY opt-in** — `karc_conformal_width` (riir-engine, +113.9% overhead per Plan 512 — FAIL default promotion), `salience_conformal_width`, 4 probe features. The three-layer split (primitive DEFAULT-ON + consumer gates opt-in) is the canonical append-only pattern. Pure modelless (empirical-quantile calibration). Zero runtime cost unless invoked. See [`.benchmarks/340_conformal_goat.md`](../../.benchmarks/340_conformal_goat.md) + [`.docs/04_calibration/conformal_predictive_intervals.md`](../04_calibration/conformal_predictive_intervals.md). |
| `karc_forecaster` | `katgpt-core/karc_forecaster` | KARC — Kolmogorov-Arnold Reservoir Computing delay-basis ridge trajectory forecaster (Plan 308, Research 288, arXiv:2606.19984 Huang/Kurths/Tang). `KarcForecaster<D,M,K>` + sealed `KarcBasis` trait (Fourier/Chebyshev/BSpline) × closed-form ridge readout. Phase 2 ships higher-order R=2 + chunked Gram + ALS low-rank `Wout ≈ A·B` (the form that persists into a `KarcShard` in riir-neuron-db). **DEFAULT-ON** (Phase 22, 2026-07-21, Issue 186 Path D3 split-config gate): G1 NRMSE **9.43e-4** PASS at K=8/M=8/R=2 d_h=18_720 (λ=5e-2 λ-sweep recovering the underdetermined system); G1 threshold **8.16 LT** PASS at K=8/M=24/R=1 λ=5e-3 (Phase 1 + Phase 5.3 confirm). Both passing configs at the same K=8 delay length; the compound gate (both legs in ONE config) is structurally infeasible — NRMSE requires R=2, threshold requires M≥24, R=2 × M=24 → d_h ≥ 166_752 (Gram ≈ 222 GB). G2 (381 ns), G3 (0 allocs), G4 (bit-reproducible Wout) ALL PASS. Pure modelless (no training, no learned params). Zero runtime cost unless a caller constructs a KarcForecaster. See [`.benchmarks/308_karc_goat.md`](../../.benchmarks/308_karc_goat.md). |
| `karc_householder_eig` | `katgpt-core/karc_householder_eig` | Issue 186 Path B (2026-07-20) — swaps the G-path eigendecomp in `low_rank_fit_jacobi_bstep` from `karc::jacobi_eigen` (O(d_h³·n_sweeps), infeasible at d_h > ~5000) to `linalg::symmetric_eig` (Householder tridiag + implicit-shift QL, ~5-10× faster at d_h ≥ 256, feasible at d_h=18_720). The new eigensolver is always compiled as a generic `linalg` primitive; this feature gates only the wiring in `karc::large_dh`. Implies `karc_forecaster`. **Opt-in** — T1-T5 PASS (correct + 7-14× faster than Jacobi at n ≤ 512); T6 (d_h=18_720 ≤30 min wall) PASS via full-rank direct Cholesky path. The Householder+QL path itself stays opt-in because direct Cholesky is both faster and more accurate for the G1 measurement. |
| `karc_householder_eig_par` | `katgpt-core/karc_householder_eig_par` | Issue 187 (2026-07-20) — row-parallel rayon variant of `linalg::symmetric_eig` for the d_h=18_720 timing trial. The four row-parallel hot loops (Householder matvec, rank-2 update, Q accumulation, QL eigenvector rotation) parallelize across rows via `par_chunk_mut(n)`; each row's work is fully sequential so the result is bit-identical to the serial path. Implies `karc_householder_eig`. **Opt-in** — T5/T6 PASS (row-parallel bit-identity + timing feasibility). **Landed a critical QL convergence fix** for near-singular Grams (the NR-local check `|e[m]| + dd == dd` cannot deflate tiny-eigenvalue matrices; added the LAPACK `dsteqr` global-scale criterion — affects both serial and parallel paths). Stays opt-in: full-rank direct Cholesky is preferred for the G1 measurement at d_h=18_720. See [`.issues/187_karc_householder_eig_parallel.md`](../../.issues/187_karc_householder_eig_parallel.md). |
| `karc_regime_gate` | `katgpt-core/karc_regime_gate` | Plan 556 Phase 1 (2026-07-20) — KARC Regime Gate. Closed-form residual-MSE mux between `KarcForecaster` (chaotic-regime specialist) and `SeasonalNaiveForecaster` (periodic-regime floor). Directly fixes the structural periodic-blindness documented in `.benchmarks/010_report_the_floor_consolidated.md` §T7 (K-sweep refuted the "K=4 too shallow" hypothesis: KARC's basis can't fit periodic data regardless of K). Two `WelfordMse` accumulators + sigmoid confidence + cold-start floor. **Revised from variance-only to MSE (variance + bias²)** after Plan 514 surfaced the failure mode where a consistently-biased forecaster has variance 0 but large error. Implies `karc_forecaster` (the gate routes to KARC) + `conformal_predictive_intervals` (the floor). **Opt-in** — primitive-level G1+G2+G3+G4 ALL PASS (37 ns median `decide()`, 0 allocs, bit-identical when gate routes to KARC); runtime-integration gain measured by riir-ai Plan 514 Phase 1: **G1 PASS (92.45% MAE reduction on mixed-regime NPC corpus)** + G2 ~at-budget (89 ns/tick). Stays opt-in pending production-corpus gain. See [`.benchmarks/556_karc_mitigations_goat.md`](../../.benchmarks/556_karc_mitigations_goat.md). |
| `karc_batched_matvec` | `katgpt-core/karc_batched_matvec` | Plan 556 Phase 2 (2026-07-20) — KARC Batched MatVec. SIMD-batched forecast across N forecasters of identical (D,M,K) shape. Crowd-scale perf primitive: amortizes memory bandwidth by laying out N `Wout` matrices contiguously and hoisting the per-output-row `simd::simd_matvec` call across the batch. `KarcBatchForecaster` + `karc_batched_matvec_into`. Implies `karc_forecaster` (operates on KarcForecaster's Wout). **Opt-in** — G1+G3+G4 PASS; **G2 PARTIAL PASS** (`.benchmarks/556_karc_mitigations_goat.md`, 2026-07-20): pure-matvec amortizes well (4.0× at N=8, 7.0× at N=32 — contiguous-layout + loop-hoisting wins); full-forecast amortization does NOT materialize (1.05× at N=8) because per-NPC `feature_expand` dominates (~75% of per-forecast cost) and is not amortized by the batched matvec. Hitting the original G2 full-forecast target requires a separate `feature_expand_batched` primitive (future work). **Plan 514 Phase 3 architecture validated:** the right consumer is cell-shared-KARC + per-NPC latent_functor deviation (ONE feature_expand per cell, batched matvec across N NPC Wouts), not per-NPC-Wout batching. Pure modelless (linear algebra only). Zero runtime cost unless constructed. Opt-in until Plan 514 Phase 3 cell-shared design demonstrates the gain. |
| `karc_lod_tier` | `katgpt-core/karc_lod_tier` | Plan 556 Phase 3 (2026-07-20) — KARC LOD Tier. Config tag + tier-promotion Wout projection. Three nested tiers (LOD0 background D=8/M=4/K=2 d_h=64 / LOD1 midground D=8/M=8/K=4 d_h=256 / LOD2 hero D=8/M=8/K=8 d_h=512) map to different `KarcForecaster` const-generic monomorphizations. Nested-subset structure (LOD0 features are a strict prefix of LOD1; LOD1 of LOD2) makes tier promotion a pure index remap — down-tier preserves surviving Wout columns bit-identically; up-tier zero-fills new columns. R=1 only in Phase 3; R=2 promotion-gate config (d_h=18_720, Issue 185/186/187) deferred. Pure modelless (matrix projection). Zero per-tick cost; tier promotion is one-time per NPC. Implies `karc_forecaster`. **Opt-in** — primitive-level GOAT G1-G4 all PASS (worst-case tier promotion 831 ns vs 10 µs target). **Runtime integration (riir-ai Plan 514 Phase 2) — honest split verdict**: G2 **PASS at 1k production scale** (14.7% savings, 5.3× headroom, re-validated 2026-07-20) but **FAIL at 10k crowd scale** (4.9% savings — dormant-Lod1 memory overhead cancels the compute savings because 10k-NPC state exceeds L3 cache, so memory bandwidth dominates). Plan 514 Phase 3/4 G2 targets revised from "10k NPCs on a single node" to "1k NPCs per shard"; the missing crowd-scale NPC sharding layer is tracked at `riir-ai/.issues/556_npc_sharding_for_crowd_scale.md`. Stays opt-in until either a pure-enum redesign (breaks `forecaster()` API) or a positive gain on a smaller-scale corpus. See [`.benchmarks/556_karc_mitigations_goat.md`](../../.benchmarks/556_karc_mitigations_goat.md) + [`riir-ai .benchmarks/514_karc_lod_dispatch_goat.md`](../../riir-ai/.benchmarks/514_karc_lod_dispatch_goat.md). |
| `poincare_navigator` | `katgpt-core/poincare_navigator` (implies `subspace_phase_gate`) | Poincaré Adapter — closed-form latent navigation distilled from SeeSE3 (Plan 449, Research 449, arXiv:2607.14228 Chen et al. DeepMind 2026). A frozen `PoincareAdapter` Pod holds `(φ, W, W†)` — given a desired movement in target space (3D pose delta / HLA affect delta), recover the latent step via `z_dest = z_src + φ⁻¹(φ(z_src) + W†·Δtarget)`. `poincare_navigate_into` is the zero-alloc hot path; `fit_poincare_adapter` is the cold closed-form ridge + PCA + thin SVD fit. **DEFAULT-ON** (Plan 449 Phase 3 promotion, 2026-07-18): G1–G7 GOAT gate ALL PASS — G1 local decodability (max |decoded delta| 0.0126, 800× under sanity bound); G3 inverse navigation Hit@0.3 = **1.000** (perfect); G4 0 allocs/100 calls; G5 `poincare_navigate_into` **809 ns/call** at d=64 (≤1µs target, 20% headroom); G6 4-step open-loop trajectory bit-identical + bounded; G7 latent-vs-raw boundary (TypeId check). G2 caveat (modelless PCA-tanh adapter R²=0.71 < linear-only ridge R²=0.93 on a coupled curved fixture) **closed by riir-train Plan 317** — trained 2-layer MLP φ reaches R²=0.9997. Promotion pattern (manifold_bandit P370 / set_attention P354 / ac_prefix P313): modelless + zero-cost-unless-invoked + GOAT-passes-on-load-bearing-axis (G3 inverse navigation, G4 zero-alloc, G5 latency) → DEFAULT-ON. Co-gates `subspace_phase_gate` for `thin_svd_into`; reuses Plan 308 `ridge_solve_direct_f32`. Pure modelless (closed-form PCA + ridge + SVD pseudoinverse). Zero runtime cost unless invoked. See [`.benchmarks/449_poincare_goat.md`](../../.benchmarks/449_poincare_goat.md) + [`.docs/05_adaptation/poincare_navigator.md`](../05_adaptation/poincare_navigator.md). |
| `chunked_content_store` | `katgpt-core/chunked_content_store` | ChunkedContentStore — Lore-distilled chunked content-addressed Merkle blob store (Plan 448, Research 262, EpicGames/lore). Bytes → `FixedSizeChunker` / `FastCdcChunker` (content-defined chunking) → BLAKE3 per chunk → dedup via `papaya` lock-free hashmap → binary Merkle root = `BlobId`. O(log n) inclusion proofs via `build_binary_merkle_proof` + light-client-friendly associated fn `verify_binary_merkle_proof` (no `&self`). `chunked_net_fetch` adds the optional `NetChunkFetcher`. **DEFAULT-ON** (Phase 19b promotion fix-up, 2026-07-18 — bench file recorded promotion but Cargo.toml entry was missed until then): G1–G7 GOAT gate ALL PASS — G1 dedup **8.47×** on 90%-shared corpus (≥5.0 target); G2 incremental push **1.35%** bytes touched (CDC) vs 52.94% (FixedSize negative control, ≤5% target); G3 inclusion prove **588 ns** + verify < 1µs (release; 2088× speedup after cached Merkle levels fix in Plan 448); G4 type-system-enforced light-client verify (associated fn, no `&self`); G5 hot-path p99 < 200 ns (release, papaya zero-alloc `.copied()`); G6 `--no-default-features` clean; G7 tamper detection **10000/10000** on 1-bit flip. Pure modelless (BLAKE3 + binary Merkle, no training, no learned params). Zero runtime cost unless a caller constructs a store. Consumed by riir-ai Plan 319 (Executable Asset Vessel + Quorum Gitflow) — `AssetStoreAdapter<S: ChunkedContentStore>` in `crates/riir-ffi/src/asset_vessel_sidecar.rs`. See [`.benchmarks/262_chunked_content_store_goat.md`](../../.benchmarks/262_chunked_content_store_goat.md) + [`.docs/03_memory/chunked_content_store.md`](../03_memory/chunked_content_store.md). |
| `smooth_min_similarity` | `katgpt-core/smooth_min_similarity` | Smooth-Min Soft Similarity — variable-length multi-token retrieval aggregator (Plan 437, Research 385, arXiv:2602.10908 SoftMatcha 2 Yoneda et al. ICML 2026). `smooth_min_similarity(cosines: &[f32], beta: f32) -> f32` interpolates between plain-min (β→∞, strictest) and plain-sum (β≈1, most lenient) — penalizes low-cosine positions more than plain mean, defeating the "distractor with 1-2 exact-match positions + several unrelated positions" failure mode. **DEFAULT-ON** (Issue 041 T6 consumer GOAT PASS, 2026-07-12): PoC GOAT G1 recall@5 **+12.0pp** (0.815 vs 0.695) on synthetic 200-item / 200-query fixture; G2 latency overhead **~0 ns** (LLVM vectorized); G3 β sensitivity all β ∈ [10¹, 10⁶] beat plain cosine. Consumer GOAT (T6 SmoothMinAligned in katgpt-attn-match): recall@5 = **1.000** vs Cosine 0.495 (+50.5pp) on position-aligned multi-token retrieval. Pure modelless (arithmetic on pre-computed cosines, no training, no weights, zero deps). Zero runtime cost unless called. |
| `octree_ctc` | `sense_composition` (alias) | OctreeCTC Reconstructive Memory Navigation (Plan 248, Research 216, arXiv:2606.06036). `octree_ctc` is an **alias feature** for `sense_composition` in katgpt-core — the standalone feature was removed from the root crate after Issue 007 Phase C moved the only consumers (`octree_ctc_demo` + recall test) to riir-engine; katgpt-core still ships the alias for direct consumers. **DEFAULT-ON** (Plan 248 Phase 5): GOAT PASS — recall ≥ 20%, **93.2 ns** < 200 ns target. Pure modelless (octree reconstruction + cosine gates). Zero runtime cost unless a caller constructs a reconstruction. |
| `sector_projection` | `katgpt-sense/sector_projection` (forwarded via `katgpt-core/sector_projection`) | SectorProjection — multi-sector spatial projection primitive (Plan 262, Research 216). `SectorProjection<N_DIR, N_SECTOR>` projects an observation onto a fixed bank of canonical sector directions — the spatial-cognition half of the Latent Physics pair (with `action_bridge`). Latent→raw bridge for NPC perception ("where am I being pushed from?"). **DEFAULT-ON** since Plan 262 Phase 2 GOAT gate. Pure modelless (closed-form dot products). Zero runtime cost unless constructed. |
| `spectral_differentiation` | `katgpt-core/spectral_differentiation` (implies `dep:rustfft`) | Spectral Differentiation — standalone FFT-based spectral differentiation for periodic uniform 1D grids (Plan 325, Research 307 §3 candidate #2, arXiv:2511.05963 *Fourier Neural Operators Explained* §2.1). The specialized case where DEC's general `exterior_derivative` (cell-complex machinery) is overkill — closed-form FFT + frequency-domain multiplier `(iω)^m`. **DEFAULT-ON** (Plan 325 Phase 3, 2026-06-25): G1 order-1 err **5.4e-7**<1e-4 + order-2 err 1.3e-6<1e-3 + spectral-vs-FD **290×** ≥100×; G2 N=1024 **3.82µs**<50µs (13× under); G3 order=0 identity bit-identical (err 2.4e-7<1e-5); G4 0 allocs/100 calls (cached `Arc<Fft>` plans + `process_with_scratch`). Pure modelless closed-form FFT. |
| `arg_protocol` | `katgpt-core/arg_protocol` | ARG Protocol Primitives — generic protocol vocabulary distilled from the ARG Standard (Plan 327, Research 309, Iris Technologies 2026). Ships: `PolicyEnvelope` + `TaxonomyValidator` (264-node) + `LifecycleState` + `RedirectTable` + `TypedOfflineCandidate` + `OfflineCandidateScorer` + `InfoRegistry`. **DEFAULT-ON** (Plan 327 Phase 4, 2026-06-25): G1 61 tests; G2a PolicyEnvelope ~0.4ns<50ns; G2b TaxonomyValidator ~170ns<200ns; G3 all-features/default/no-default clean; G4 0 allocs/100 calls (fixed via scratch + clone-instead-of-mem::take); G5 silence-bias strict inequalities. Pure modelless protocol vocabulary — no game/chain/shard IP. Composes with `non_interference_branches` `LifecycleState` when both features on. Private runtime wiring: riir-ai Plan 337 / Guide 160. |
| `phase_rotation_coupling` | `katgpt-core/phase_rotation_coupling` | Phase-Modulated Coupling — norm-preserving subspace rotation gate (Plan 322, Research 305, arXiv:2605.12700 UFO Qiao/Karniadakis/Munirazzaman May 2026). `cos α ⊙ a + sin α ⊙ b` where α comes from a sigmoid projection — the open math hook for norm-preserving NPC affect rotation / crowd-coherent mode transition / chain-committed phase for deterministic replay. **DEFAULT-ON** (Plan 322 Phase 2, 2026-06-25): G1 per-channel Pythagorean drift **5.96e-8**<1e-4 (1677× headroom); G2 0 reversals/100-step sweep (monotone interpolation); G3 D=8 scalar+mix **18.9ns**<50ns + D=8 mix-only **5.0ns**<20ns + D=64 per-channel+mix 355.7ns<1500ns; G4 0 allocs; G6 sigmoid(0)=0.5→cos=sin=1/√2 (softmax would give 1.0). Design pivot: independent Padé cos/sin drifts in cos²+sin²=1 by ~5e-3 (50× G1 budget) — replaced with `phase_safe_cos_sin` (libm sin + Pythagorean sqrt(1−sin²) recovery). Pure modelless. Private fusion guides deferred to riir-ai (HLA runtime) / riir-chain (LatCal committed phase) / katgpt-rs (DEC Hodge mixer). |
| `non_interference_branches` | `katgpt-core/non_interference_branches` | Non-Interference Memory Branches — continual adaptation primitive distilled from RIZZ (Plan 329, Research 310, arXiv:2606.20638 Goel et al. Oxford Jun 2026). Five generic primitives: `BranchBank` + `BranchRouter` + `VerifierGate` + `NonInterferenceProjection` + `BudgetCompiler`. The Super-GOAT fusion of BAKE × CLR × MCGS × Engram × ARG × closure-instrument × Salience into per-NPC continual adaptation without catastrophic forgetting. **DEFAULT-ON** (Plan 329 Phase 3, 2026-06-26): G1 8 orthogonal directions in D=8 (pairwise interference 0.00e0<1e-6; write to b_i does not contaminate b_j stores; 9th direction correctly rejected at 0.3536≥1/√8); G2 route **301.5ns**<1µs (64-branch bank, 3.3× margin); G3 all-feature combos clean; G4 0 allocs/100 calls; G5 `[]` deps. 101/101 unit tests. Composes with `arg_protocol` LifecycleState when both features on. Pure modelless (structural geometric orthogonality, not learned). Private runtime wiring deferred to riir-ai Plan 338. |
| `best_belief` | `katgpt-core/best_belief` | Best-Belief Beta Selector — ε-quantile Beta lower bound for conservative selection (Plan 336, Research 320, RQGM arXiv:2606.26294 Prop. 4). Complements `sample_beta` (Thompson sampling for EXPLORATION) with a conservative EXPLOITATION/SELECTION counterpart. **DEFAULT-ON** (Plan 336 Phase 2 G2-unblock, 2026-06-28): LUT hot path **3.38ns**, G1 3.099e-5<1e-4 vs statrs, G4 0 allocs. **Issue 010 T5 "Report the Floor" comparison: BEATS the MLE floor** in the heteroscedastic regime (variable observation counts — the real-world use case for frozen snapshots/archetype shards with different deployment durations → 15–30% selection regret ↓); ties at uniform n (the monotonicity theorem). Confirms DEFAULT-ON promotion. Pure modelless (closed-form Beta inverse-CDF via LUT). |
| `cognitive_architecture_root` | `katgpt-core/cognitive_architecture_root` (implies `engram`) | Cognitive Architecture Root — whole-architecture BLAKE3 commitment `CognitiveArchitectureRoot([u8; 32])` (Issue 039, 2026-07-04). The anti-cheat / quorum-attested personality freeze-thaw / on-chain NPC avatar portability primitive. Implies `engram` (so `engram` is transitively default-on via this feature — the Plan 299 "default-off" label predates this promotion, see Plan 360 status sync note). **DEFAULT-ON** (Issue 039, 2026-07-04): G1 spec-match 13/13 + bit-flip every input; G1-avalanche min 120/256 avg 126/256 (BLAKE3 ~128, floor 96); G2 `from_parts` 208ns + `verify` 208ns (<500ns); G2-alloc 0/1000; G3 `--all-features` + `--no-default` clean; G4 `size_of == 32`. Pure modelless. Zero runtime cost unless a caller constructs/verifies a root. |
| `ptg_functor_edges` | `katgpt-core/ptg_functor_edges` (implies `closure_instrument`) | PTG × latent_functor Edge composition (Issue 040, 2026-07-04). Adds `FunctorPtg` composite (wraps an unchanged `PrimitiveTransitionGraph` with a parallel `Vec<Option<FunctorEdgeParams>>`) + `apply_functor_edge_into` (zero-alloc sigmoid-gated cosine·direction apply path) + `functor_edge_gate` (diagnostic gate query). Wire-format safe: the inner PTG is byte-identical to a bare PTG (T1 audit found postcard `#[serde(default)]` does NOT work for missing trailing fields — "Hit end of buffer" — so the composite approach is mandatory). Implies `closure_instrument`. **DEFAULT-ON** (Issue 040 T7, 2026-07-04): G1 6/6 sub-checks (high-coherence ≈ state+dir, low-coherence ≈ identity, determinism, threshold gate=0.5, FunctorPtg preserves inner commitment, wire-format byte-identical) + 17 unit tests; G2 `apply_functor_edge_into` **28.5ns** at D=64 (target <200ns, 7× headroom); G2-alloc 0/1000; G3 default + `--all-features` + `--no-default` clean; G4 `size_of::<FunctorEdgeParams> == 44` bytes (no heap indirection); G5/G6 pure modelless (closed-form cosine + sigmoid + SAXPY). |
| `heal_validation` | `katgpt-core/heal_validation` | Heal-Validation Conflict Detector — `HealConflictDetector` trait for healed-state semantic validation (Issue 133, 2026-07-12). The heal-path analog of LDT's `ConflictDetector` (Plan 088): where `ConflictDetector` checks token candidate sets for satisfiability (signature: `marginals`, `pruned_count`, `total_candidates`), this checks healed flat `&[f32]` state (style_weights for shards, emotion axes for HLA) for semantic impossibility (NaN, degenerate blend, anger+calm both >0.7, etc.). The signature is intentionally different — forcing the token-specific signature onto heal validation would abuse its parameters (Interface Segregation Principle). **Passive trait** — zero behavior change unless consumers implement it. **DEFAULT-ON** (Issue 133, 2026-07-12): G1–G6 ALL PASS. Two consumer impls pass GOAT: `ShardConflictDetector` (riir-neuron-db, **30ns**) and `HlaConflictDetector` (riir-games, **2ns**), both <50ns target. Pure modelless (threshold checks). |
| `full` | all above (excludes `stepcode`, `sp_kv`, `shard_kv`, `peira_distill`, `dirichlet_energy`, `data_probe`, `rmsd_distill`, `safe_bandit`, `stiff_anomaly`, `state_source`, `nexus_elo`, `skill_opt`, `proof_cert`, `mech_attribution`, `ega_attn`, `event_log`, `spec_cost_model`, `spechop`, `rt_turbo`, `tf_loop`, `plasma_path`, `parallel_probe`, `parallax_attn`, `sigmoid_margin`, `moa_inference`, `dual_gram_pca`, `roofline_cost`, `leo_all_goals`, `dual_leo`, `stability_metrics`, `asymmetric_kv`, `kog_cpu_fusion`, `caddtree_budget`, `sense_composition`, `bake_precision`, `induced_cwm`, `induced_cwm_ismcts`, `induced_cwm_tournament`, `interpolation_geometry`, `grapem_rodrigues`, `position_group_action`, `grape_ap_vector`, `grape_joint_lift`) | Enable all features |

Default features: `sparse_mlp`, `domain_latent`, `ppot`, `bandit`, `bandit_top_p`, `bt_rank`, `spectral_quant`, `hybrid_oct_pq`, `elf_sde`, `cna_steering`, `deep_manifold`, `federation`, `tes_loop`, `lattice_deduction`, `delta_routing`, `stability_metrics`, `mls_aggregate`, `gdn2_attention`, `dash_attn`, `dreamer`, `lt2_looped`, `dmax_spd`, `eqr_convergence`, `subterranean`, `sr2am_configurator`, `data_gate`, `plasma_path`, `parallel_probe`, `tf_loop`, `leo_all_goals`, `dual_leo`, `sigmoid_margin`, `moa_inference`, `sleep_consolidation`, `spectral_hierarchy`, `dual_gram_pca`, `roofline_cost`, `newton_schulz`, `river_valley`, `peira_distill`, `kog_cpu_fusion`, `gepa_reflective`, `phrase_boost`, `hydra_budget`, `budget_adaptation`, `ilc_distill`, `thinking_prune`, `rim_slots`, `thinking_cot`, `freq_bandit`, `spec_reconciliation`, `trust_region_spec`, `curvature_alloc`, `directional_credit`, `kv_share`, `nds_proxy`, `wealth_pruner`, `speculative_generator`, `kvarn`, `and_or_dtree`, `belief_drafter`, `bfcf_lfu_shard`, `slod`, `schema_centroid`, `union_bound_confidence`, `pathway_tracker`, `federation_composer`, `llmexec_guard`, `outlier_guard`, `segment_checkpoint`, `self_distilling_bandit`, `precision_aware_draft`, `static_cal_tables`, `targeted_precision`, `egcs`, `reward_mem`, `symbolic_distill`, `concept_grounding`, `reward_calibrator`, `decision_explain`, `collapse_aware_thinking`, `temporal_deriv`, `ilc_distill`, `kog_cpu_fusion`, `clr`, `viable_manifold_graph` (implies `subspace_phase_gate`), `ac_prefix`, **`manifold_power_iter_router`** (Plan 279 Phase 4 GOAT 9/9 — composed pipeline), **`quantile_balance_router`** (Plan 455 Phase 3 Case C 2026-07-17 — composed with `manifold_power_iter_router` as the recommended snapshot-swap reconditioning), **`causal_identification`** (Plan 457 Phase 5 promotion 2026-07-18 — Cakiqi-Little syntactic causal ID on ADMGs; G4 alloc-free closed by Issue 183 + Issue 184 / Bench 466, allocs 284→133/call −53% cumulative, latency 8.26→6.07µs), **`conformal_predictive_intervals`** (Plan 468 promotion 2026-07-20 — modelless conformal UQ overlay + canonical Issue 010 "Report the Floor" SeasonalNaive m=1 instance; primitive G1–G4 PASS Bench 340: coverage [0.9445, 0.9493], `interval_into` H=1 642ns, 0 allocs; runtime-consumer gate satisfied by Bench 564 MCTS collapse G3 + Bench 565 Salience Tri-Gate G3 dF1=+0.3145; consumer gates stay opt-in per the three-layer split pattern), **`poincare_navigator`** (Plan 449 Phase 3 promotion 2026-07-18 — closed-form latent navigation distilled from SeeSE3 arXiv:2607.14228; G3 inverse Hit@0.3=1.000, `poincare_navigate_into` 809ns ≤ 1µs target, 0 allocs steady-state; G2 caveat modelless PCA-tanh R²=0.71 < linear ridge 0.93 closed by riir-train Plan 317 trained φ R²=0.9997; promotion pattern matches manifold_bandit P370 / set_attention P354 / ac_prefix P313 — modelless + zero-cost-unless-invoked + GOAT-passes-on-load-bearing-axis → default-on), **`chunked_content_store`** (Plan 448 + Phase 19b promotion fix-up 2026-07-18 — Lore-distilled content-addressed Merkle blob store; G1 dedup 8.47×, G2 CDC incremental push 1.35% vs FixedSize 52.94%, G3 prove 588ns + light-client verify < 1µs via cached Merkle levels, G7 tamper detection 10000/10000; consumed by riir-ai Plan 319 Asset Vessel), **`smooth_min_similarity`** (Issue 041 T6 consumer GOAT PASS 2026-07-12 — smooth-min soft similarity for multi-token retrieval distilled from SoftMatcha 2 arXiv:2602.10908; PoC +12.0pp recall@5, consumer +50.5pp; ~0ns overhead, pure modelless arithmetic on pre-computed cosines), **`octree_ctc`** (Plan 248 — alias for `sense_composition` in katgpt-core; 93.2ns<200ns, recall ≥20%), **`sector_projection`** (Plan 262 — multi-sector spatial projection, Latent Physics pair half with `action_bridge`), **`spectral_differentiation`** (Plan 325 Phase 3 2026-06-25 — FFT-based spectral differentiation for periodic 1D grids, arXiv:2511.05963; G1 spectral-vs-FD 290×, G2 N=1024 3.82µs<50µs, G3 order=0 identity, G4 0 allocs), **`arg_protocol`** (Plan 327 Phase 4 2026-06-25 — ARG Standard protocol primitives: PolicyEnvelope+TaxonomyValidator+LifecycleState+RedirectTable+TypedOfflineCandidate+OfflineCandidateScorer+InfoRegistry; G1 61 tests, G4 0 allocs), **`phase_rotation_coupling`** (Plan 322 Phase 2 2026-06-25 — norm-preserving subspace rotation gate `cos α ⊙ a + sin α ⊙ b`, arXiv:2605.12700 UFO; G1 drift 5.96e-8<1e-4, G3 D=8 18.9ns<50ns, G4 0 allocs, G6 sigmoid-not-softmax), **`non_interference_branches`** (Plan 329 Phase 3 2026-06-26 — RIZZ distillation arXiv:2606.20638; BranchBank+BranchRouter+VerifierGate+NonInterferenceProjection+BudgetCompiler; G1 8 orthogonal dirs in D=8, G2 301.5ns<1µs, 101/101 tests), **`best_belief`** (Plan 336 Phase 2 2026-06-28 — ε-quantile Beta lower bound for conservative selection, arXiv:2606.26294 RQGM Prop. 4; LUT hot path 3.38ns; Issue 010 T5 BEATS MLE floor in heteroscedastic regime), **`cognitive_architecture_root`** (Issue 039 2026-07-04 — whole-architecture BLAKE3 commitment; implies `engram`; G1 spec-match 13/13, G2 208ns<500ns, G4 size_of==32), **`ptg_functor_edges`** (Issue 040 T7 2026-07-04 — PTG × latent_functor edge composition; implies `closure_instrument`; G1 6/6 sub-checks + 17 unit tests, G2 28.5ns<200ns at D=64, G4 size_of==44), **`heal_validation`** (Issue 133 2026-07-12 — `HealConflictDetector` trait for healed-state validation, heal-path analog of LDT's ConflictDetector; passive — zero behavior change unless consumers implement; consumer impls ShardConflictDetector 30ns / HlaConflictDetector 2ns, both <50ns target; G1–G6 ALL PASS). (~98 default features — production best perf + accuracy, all GOAT-proved, Plans 051–468.)

## Quick Start

```bash
cargo test --quiet --workspace --all-features   # Run all 740+ tests
cargo run --release                             # Run benchmark suite (includes Leviathan verification)
cargo run --example sudoku_01_9x9 --features sudoku           # Sudoku streaming solver
cargo run --example sudoku_02_speculative --features sudoku   # DDTree pruning demo
cargo run --example sudoku_03_tui --features sudoku           # TUI visualization
cargo run --example core_01_validator --features validator     # SynPruner + DDTree pipeline
cargo run --example core_02_raven                             # Raven RSM demo
cargo run --example core_03_ppot --features ppot              # PPoT resampling demo
cargo run --example core_04_prefill                           # PFlash prefill demo
cargo run --example bandit_01_basic --features bandit         # Bandit basics
cargo run --example bomber_01_arena --features bomber         # Bomberman arena
cargo run --example bomber_09_rubric_tournament --features ropd_rubric,g_zero,bomber  # Bomber rubric tournament (Plan 076)
cargo run --example monopoly_01_arena --features monopoly     # Monopoly arena
cargo run --example fft_01_arena --features fft               # FFT Tactics arena
cargo run --example fft_02_rubric_tournament --features ropd_rubric,g_zero,fft  # FFT rubric tournament (Plan 076)
cargo run --example go_06_bench --features go --release       # Go benchmark suite
```

## Config Presets

| Config | vocab | embd | heads | layers | mlp | Purpose |
|--------|-------|------|-------|--------|-----|---------|
| `micro` | 27 | 16 | 4 | 1 | 64 | Default benchmark target |
| `micro_lora` | 27 | 16 | 4 | 1 | 64 | Micro + LoRA adapter support |
| `draft` | 27 | 4 | 2 | 1 | 16 | Tiny draft model |
| `game` | 27 | 16 | 4 | 1 | 64 | Game domain preset (domain_latent) |
| `bpe` | 4096 | 32 | 4 | 1 | 128 | BPE Rust code model |
| `bpe_draft` | 4096 | 8 | 2 | 1 | 32 | BPE draft model |
| `small_target` | 4096 | 64 | 4 | 4 | 256 | Multi-layer target |
| `gqa_draft` | 4096 | 64 | 8 | 4 | 256 | GQA draft (n_kv_head=2) |
| `micro_dllm` | 27 | 16 | 4 | 1 | 64 | D2F discrete diffusion (bidirectional) |
| `game_go` | 85 | 32 | 4 | 1 | 128 | Go board 9×9 + action (~16K params) |
| `qwen_deltanet` | 151936 | 2048 | 16 | 4 | 8192 | QwenDeltaNet hybrid DeltaNet/Attention (kv_heads=8, head_dim=128, Plan 182) |
| `gemma2_2b` | 256000 | 2304 | 8 | 26 | 9216 | Gemma 2 2B architecture (kv_heads=4, head_dim=256) |

### ManifoldPruner Code Example (Plan 234, opt-in)

```rust
// Before: Binary pruning (misses boundary tokens)
if pruner.is_valid(depth, token, prefix) {
    tree.expand(token);
}

// After: ManifoldPruner captures boundary tokens
if pruner.manifold_score(depth, token, prefix) > threshold {
    tree.expand(token); // threshold < 0.5 captures boundary tokens
}
```

> **Note:** G1 FAIL — `sigmoid(x) > 0.5 ⟺ x > 0`, so at default 0.5 cutoff this is identical to binary pruning. The Gaussian kernel (G2 PASS) remains valuable for ranking.

## Key Design Principles

1. **Zero allocations on hot paths** — all buffers pre-allocated in `SpeculativeContext` and `ForwardContext`
2. **Feature-gated modularity** — domain code (sudoku, validator) never pollutes core
3. **Trait-based strategy** — `ConstraintPruner`, `SpeculativeVerifier`, `PrefillScorer`, `ScreeningPruner` for swappable behavior
4. **SOLID module decomposition** — each file < 1024 lines, single responsibility
5. **`mod.rs` for index only**, minimal `main.rs`/`lib.rs`
6. **Unsafe only in verified hot-path kernels** with `get_unchecked` + `#[inline(always)]` + SIMD intrinsics (`core::arch` NEON/AVX2)

## Related Documentation

Docs are grouped into topic folders (no number prefix) — see [`.docs/README.md`](../README.md)
for the full index with fusion maps. Quick map:

| Group | Docs | Topic |
|---|---|---|
| [`orientation/`](../01_orientation/) | `overview.md` | Overview & reference card (this file) |
| [`orientation/`](../01_orientation/) | `architecture.md` | Architecture details (forward pass, routers, LoRA) |
| [`orientation/`](../01_orientation/) | `paper_feature_comparison.md` | Paper feature comparison |
| [`inference/`](../02_inference/) | `speculative_decoding.md` | Speculative decoding deep-dive |
| [`inference/`](../02_inference/) | `spechop.md` | SpecHop architecture |
| [`inference/`](../02_inference/) | `kv_compression.md` | KV compression alternatives |
| [`inference/`](../02_inference/) | `mtp_threshold.md` | MTP threshold guide (Plan 055) |
| [`inference/`](../02_inference/) | `progressive_mcgs.md` | Progressive MCGS graph search |
| [`memory/`](../03_memory/) | `raven_rsm.md` · `product_key_memory.md` · `engram.md` · `micro_belief.md` · `sense_composition.md` · `sleep_consolidation.md` | Memory primitives |
| [`calibration/`](../04_calibration/) | `cce_moderator.md` · `causal_head_importance.md` · `faithfulness_probe.md` · `salience_tri_gate.md` · `universality_class_escape.md` | Calibration, probes, gates |
| [`adaptation/`](../05_adaptation/) | `model_adaptation.md` · `lucebox_techniques.md` · `peira_distillation.md` | Model adaptation & distillation |
| [`game_arenas/`](../06_game_arenas/) | `sudoku.md` · `heuristic_learning.md` · `bomber_arena.md` · `monopoly_fsm.md` · `fft_arena.md` · `go_arena.md` · `hl_arena_detail.md` · `open_ended_evolution.md` · `bomber_lora_ab.md` | HL game arenas |
| [`validator/`](../07_validator/) | `constraint_validator.md` · `percepta.md` | Constraint validator + SynPruner, transformer-VM |
| [`performance/`](../08_performance/) | `engineering.md` | Performance engineering & benchmarks |
| [`feature_catalog/`](../09_feature_catalog/) | `opt_in_features.md` · `negative_results.md` | Opt-in features, negative results |
| [`audits/`](../10_audits/) | `loser_sweep_audit.md` · `claim_rubric_audit.md` · `cross_repo_consolidation_audit.md` | One-off audits |