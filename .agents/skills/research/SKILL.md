---
name: research
description: Research workflow for distilling ML/AI papers into modelless inference primitives, freeze/thaw runtime patterns, latent-space operations, AND model-based training plans across the katgpt-rs / riir-ai / riir-chain / riir-neuron-db / riir-train / riir-game-sdk / riir-armageddon 7-repo stack. Use when reading arxiv papers, deciding which repo a paper belongs in, creating .research/ notes or .plans/ files, implementing modelless inference primitives, or routing training-vs-inference insights. Enforces the 7-repo commercial strategy (public engine / private runtime / private chain / private neuron-db / private training / private SDK facade / private product-domain), modelless-first constraint (with model-based track actively pursued as of 2026-08-06 — modelless near its limit), latent-to-latent preference, and freeze/thaw-over-fine-tuning rule.
---

# Research Workflow — Modelless Inference, Freeze/Thaw, Latent-to-Latent

Training-method research lives in `riir-train`. This repo (`katgpt-rs`), `riir-ai` (freeze/thaw runtime + self-learn/adaptive NPCs + game systems), `riir-chain` (neuro-symbolic chain transport, LatCal, chain economics), and `riir-neuron-db` (neuron weight shards, BLAKE3/Merkle commitment, freeze/thaw envelope, consolidation, AnyRAG gateway, vibe KG triples) ship **runtime + latent-space operations**. No LoRA training, no adapter fine-tuning, no optimizer research here. If a paper's value is its training loop → `riir-train/.research` **+ a Plan in `riir-train/.plans/` if applicable to our model-based track (as of 2026-08-06: training efficiency is actively pursued, not lazily redirected — see §3.5 Path 0.5)**. If its value is a latent-space insight, a routing trick, a freeze/thaw pattern, a chain-commitment bridge, a neuron-shard primitive, or a modelless inference primitive → distill here.

## When to use this skill

Activate when the user (or you) are doing any of:

- Reading / fetching / summarizing an ML, AI, or systems paper (arxiv, PDF, blog).
- Deciding which of the 7 repos a paper or idea belongs in (katgpt-rs / riir-ai / riir-chain / riir-neuron-db / riir-train / riir-game-sdk / riir-armageddon).
- Creating a new `.research/NNN_*.md` note or `.plans/NNN_*.md` plan.
- Implementing a modelless inference primitive (pruner, bandit, router, speculative decode, KV-cache op, sparse attention, quantization-aware inference).
- Designing freeze/thaw snapshot cycles, adapter hot-swap, or runtime adapter routing.
- Designing latent-to-latent operations (dot-product projection, sigmoid gating, manifold geometry, spectral methods on activations).
- Designing MMORPG-scale game AI (thousands of concurrent NPCs, 20Hz tick, fog-of-war, zone attention, emergent social/economic behavior).

Do NOT activate for: pure refactor tasks, bug fixes with no research angle, or ordinary feature work that doesn't touch the research/plans folders.

## Repos (siblings under the same parent)

- `katgpt-rs/` — public MIT engine. Generic modelless inference primitives. **No game IP, no chain IP, no neuron-shard IP.**
- `riir-ai/` — private game product. Freeze/thaw runtime, self-learn, game systems. Hosts the consolidated `.docs/` book (selling points / moats — see §Read first).
- `riir-chain/` — private neuro-symbolic chain transport. LatCal (Lattice Calculus), `riir-chaind` daemon, chain economics, Solana-parity features, asset lifecycle / forensic fingerprinting, `catchup/` (Turso/libSQL persistence, quorum), `DataTier` / `DATA_TIERS` / `build_tier_root`. **The sync-boundary bridge repo. Re-exports `riir-neuron-db` under its `neuron_db` feature, but the canonical shard source is `riir-neuron-db/`.**
- `riir-neuron-db/` — private leaf crate. `NeuronShard` (`#[repr(C)]` Pod, zero-copy mmap), `ShardIndex` (lock-free `papaya::HashMap`), generic `MerkleTree`/`MerkleProof`, MAPE-K self-healing loop, Raven/δ-Mem consolidation, AnyRAG escalation gateway, vibe KG triple templates + arch agent, `MerkleFrozenEnvelope` (freeze/thaw integrity), spectral initialization, `ShardCompactor`, dendritic LoRA branch view. **No chain dependency — usable standalone.**
- `riir-train/` — private training vault. Adapter training, optimizers, loss functions. **As of 2026-08-06: no longer "out of scope" — the modelless track is near its limit and the model-based track is actively pursued.** Training-efficiency papers applicable to our model-based track MUST get a Plan in `riir-train/.plans/` per §3.5 Path 0.5, NOT a lazy one-line redirect. Only genuinely out-of-scope training (e.g., image-specific DiT architecture we'll never train) gets the one-line redirect with explicit justification. **The model-based track is broader than Plan 318 SFT (GDN-blog lesson 2026-08-07): it includes Plan 059 (G-Zero GRPO/DPO), Plan 066 (`distill_attention.rs` SDPA→HLA), Plan 501–505 (trajectory collection). `read_file riir-train/.docs/02_pipelines/training_data_pipeline.md` before any training-paper verdict.**

**Routing rule of thumb (chain vs neuron-db):** if the mechanism is about *how a shard is structured, committed, frozen, consolidated, retrieved, or projected* → `riir-neuron-db`. If it is about *how a shard is committed to a chain block, transported across quorum, or bridged to LatCal fixed-point* → `riir-chain`. The `LatCalWalletExt` trait (typed wallet accessors on `NeuronShard` using `LatCalMatrix`) stays in `riir-chain` because it is the bridge.

Always reference files with project-relative paths (e.g. `katgpt-rs/.research/238_*.md`, `riir-ai/.plans/NNN_*.md`, `riir-chain/.plans/NNN_*.md`, `riir-neuron-db/.plans/001_*.md`). The agent can `read_file` these directly.

## Commercial strategy — inline short version

Seven repos (see §Repos). Tier model:

| Tier | Where | Role |
|------|-------|------|
| **0 — Substrate** | `katgpt-core` (leaf, crates.io) | Pure inference mechanics (SIMD, transformer/weights, `hla`, `dd_tree`, `mcts`, `sampling`, `delta_mem`). Leaf-clean deps. |
| **1 — Engine + cognitive basics** | `katgpt-rs` (root, public) | Adoption funnel — re-exports substrate + BASIC cognitive/reasoning primitives + toy games (each ships WITH its `.md`). |
| **2 — GOAT + IP** | `riir-ai` / `riir-chain` / `riir-neuron-db` / `riir-train` (private) | GOAT/Super-GOAT tuned versions, `*_runtime` composition layers, game/chain/shard/training IP. |

**Routing rules:**
- Module → core only if pure inference substrate (no heavy deps, no cognitive semantics). `hla` qualifies; `cce`/`cgsp` do not.
- `*_runtime` suffix = private GOAT composition layer. Bare-name = public primitive.
- **What = public. How = private.** Training how → `riir-train`. Runtime how → `riir-ai`. Chain how → `riir-chain`. Shard how → `riir-neuron-db`.
- When unsure → default private. Safe to keep private; never safe to un-leak public.
- **Benchmark exception:** toy 2D games (`bomber`, `monopoly`, `go`, `fft`) are NOT product IP — live in `katgpt-rs`. Test: *"Could a competitor re-implement from public rules in a weekend?"* → public.
- **Cognitive moat:** basic primitives public ("good enough to adopt"); GOAT-tuned versions private ("the version that actually wins"). `katgpt-rs` is the open Ferrari — the gas is in `riir-ai`/`riir-chain`.
- **Anti-pattern:** a public benchmark constant naming a private module path (`must match riir_gpu::...`) IS a leak — cross-boundary coupling constants forbidden.
- **FV moat:** ~79 Lean 4 theorems across 4 `.proofs/` instances. Theorems encode invariant shape; private proofs stay private. FV is bug-finder + refactor-immune guarantee.

## Read first (grounding) — MANDATORY pre-flight

**Hard rule:** before any distillation/verdict/file creation, do **all four**:
1. **`read_file` 5 READMEs + `riir-ai/.docs/README.md`** — defines repo purpose, moat map, raw-vs-latent sync boundary, AND the training scope boundary (riir-train/README.md — what trains vs what stays runtime). Skipping = #1 cause of false Super-GOAT claims AND false PASS on training papers (the EG-FM lesson, arXiv:2608.05811, 2026-08-08: an agent PASS'd a training paper without reading `riir-train/README.md`, didn't know the full training-methods inventory, and had to be pushed back by the user to re-check).
2. **`list_directory` all 5 `.research/` folders** (katgpt-rs, riir-ai, riir-chain, riir-neuron-db, **riir-train** — create `riir-chain/.research/` + `riir-neuron-db/.research/` on first use; `riir-train/.research/` already exists with 100+ training-method notes).
3. **`list_directory` 4 runtime/chain/neuron-db src trees** — module names are codebase vocabulary. Skipping = #2 cause of false Super-GOAT claims.
4. **`web_search` for published prior art** on the paper's headline technique (see §4 for the full protocol). Skipping = #4 cause of false novelty claims (the MoTE lesson, Research 411, 2026-08-09: an agent claimed "ternary MoE is novel" without searching the web for "mixture of ternary experts" — MoTE, arXiv:2506.14435, published exactly that technique in June 2025. The user had to manually prompt "search web for more paper" to catch it.)

**Reads:** `katgpt-rs/README.md`, `riir-ai/README.md`, `riir-chain/README.md`, `riir-neuron-db/README.md`, **`riir-train/README.md`** (the 5th README — defines the Issue 004 scope boundary: adapter training methods / optimizers / losses / DPO/GRPO/SFT/RL pipelines MOVE here; freeze/thaw runtime / adapter routing / game+chain+validator+inference engines STAY in riir-ai; the producer/consumer contract is train here → `lora.bin` (BLAKE3) → freeze/thaw consume in riir-ai), `riir-ai/.docs/README.md` (+ `03_pillars/README.md` + `04_supergoat_candidates/README.md` before any Super-GOAT gate — claiming novelty over a moat that ships is the worst false-positive). **MANDATORY for training-paper classification (the GDN-blog lesson, 2026-08-07 + the EG-FM lesson, 2026-08-08):** also `read_file` `riir-train/.docs/02_pipelines/training_data_pipeline.md` + `riir-train/.docs/02_pipelines/gpu_training.md` + `riir-train/.docs/01_orientation/training_topology.md` — these define the ACTUAL model-based track (Plan 423 Gemma-2-2B LoRA SFT [PRODUCTION], Plan 318 SFT/LoRA [retired], Plan 059 G-Zero GRPO/DPO, Plan 066 HLA distillation, Plan 501–505 trajectory collection). Skipping these = false PASS on training papers by narrowing 'model-based track' to just Plan 318 SFT. The model-based track is broader than one plan.

**`list_directory`:**
- 5× `.research/` (katgpt-rs, riir-ai, riir-chain, riir-neuron-db, **riir-train**)
- `riir-ai/crates/riir-engine/src/` — runtime vocab (`latent_functor/`, `cgsp_runtime/`, `hla/`, ...). Skipping caused the DiPOD miss: `latent_functor/reestimation.rs` ships "drift-triggered self-healing swap" as "coherence-driven re-estimation scheduler".
- `riir-ai/crates/riir-games/src/` — game systems
- `riir-chain/src/` — `encoding/` (LatCal), `consensus/`, `economics/`, `asset_lifecycle/`, `forensic/`, `catchup/`, ...
- `riir-neuron-db/src/` — `shard.rs`, `index.rs`, `merkle.rs`, `freeze.rs`, `mape_k.rs`, `consolidation.rs`, `gateway.rs`, `vibe.rs`, `spectral_flatness.rs`, `shard_compactor.rs`
- **`riir-train/crates/riir-train-gpu/src/` + `riir-train/crates/riir-train-engine/src/`** — training vocab (`distill_attention.rs`, `loss_grpo.rs`, `loss_dpo.rs`, `delta_filter.rs`, optimizer kernels). Skipping = false PASS on training papers (the GDN-blog lesson: an agent PASS'd a paper touching HLA distillation + GRPO without knowing these files existed).

Do NOT create any file until all four done. Plans (`.plans/`) are grepped during fusion search (§1), not pre-flight.

## Primary focus (distill HERE in katgpt-rs / riir-ai)

**Fusion-first mindset:** The highest-value Super-GOATs in this codebase come from **fusing 2–3 papers/primitives into a novel combination**, not from direct-mapping a single paper. Always grep `.research/` + `.plans/` for the 2–3 closest cousins before verdict, and ask: "what does paper × note A × note B produce that none of them alone can?" Examples that shipped: Gemini Fourier × LatCal (research 212 → plan 242); EGA × SpectralQuant (research 100 × 039); collapse-aware × bandit × sigmoid-margin (plans 212 × 157 × 061). See §Workflow step 1 for the full fusion protocol.

- **Latent-to-latent operations** — anything that stays in embedding/latent space: dot-product projections, cosine similarity retrieval, sigmoid-gated routing, manifold geometry, spectral methods on activations. Prefer operating on latents over decoding to tokens then re-encoding. **Fusion hook:** combine with freeze/thaw to version latent-direction vectors; combine with self-learn to update direction vectors from runtime curiosity signal.
- **Freeze/thaw patterns** — versioned weight snapshots, atomic hot-swap, lock-free read paths, BLAKE3/commitment-checked adapter reload, per-entity personality divergence via snapshot versioning. **Fusion hook:** combine with runtime adapter routing to dispatch by latent-state similarity; combine with self-learn to snapshot emergent NPC personalities.
- **Runtime adapter routing** — selecting between frozen adapters by state/objective/context (Dynamic Pair, Polytope, dMoE — all inference-time, zero training). **Fusion hook:** combine with freeze/thaw to make the adapter pool itself versioned and BLAKE3-committed; combine with bandits to learn routing policy online.
- **Self-learn / adaptive CoT** — runtime curiosity, entropy-driven exploration, collapse detection/recovery, latent prediction SSL, trajectory folding. No LLM training, no backprop through weights — runtime self-improvement via latent-space updates is welcome. **Fusion hook:** combine with MMORPG-scale game AI to give thousands of NPCs independent curiosity/entropy signals; combine with freeze/thaw to checkpoint learned latent directions.
- **Modelless inference primitives** — ConstraintPruners, bandits, DDTree, speculative decode, sparse attention, quantization-aware inference.
- **MMORPG-scale game AI** — thousands of concurrent NPCs each with independent latent state, real-time latency budgets (20Hz tick, plasma/hot tier), spatial partitioning + fog-of-war, emergent social/economic behavior (factions, trade routes, reputation), zone-level attention routing, crowd-scale curiosity/exploration signals. Latent ops must batch across many entities; raw sync must stay bit-identical for deterministic replay/anti-cheat.

### Super-GOAT factory modules — grep FIRST, explicitly

The highest-value latent-space Super-GOATs cluster in seven module trees. When grepping for fusion cousins and prior art, `list_directory` these explicitly — do NOT rely on keyword grep alone (vocabulary mismatch is the #3 cause of false verdicts):

| Module | What ships | Super-GOAT angle |
|---|---|---|
| `katgpt-rs/crates/katgpt-core/src/sense/` | belief-state kernels, `evolve_belief`, `SenseModule::project`, ternary bit-plane projection | Per-NPC recurrent latent state — the runtime substrate for any "hidden state" / "belief" / "activation" paper |
| `riir-ai/crates/riir-engine/src/latent_functor/` | `zone_gating.rs`, `reestimation.rs`, `arithmetic.rs`, `cross_game.rs`, `k_selector.rs`, `quality_gate.rs` | **Game-theory in latent space** — functors as vector ops, coherence-driven re-estimation, zone-gated activation. Maps any "stage" / "application" / "bypass" / "collapse" paper |
| `riir-ai/crates/riir-engine/src/hla/` | `kernel.rs`, `forward.rs`, `types.rs` — **Higher-order Linear Attention** (Transformer attention layer replacement; re-exports `katgpt-hla` kernels + adds `*_role_aware` variants behind `hla_role_aware` feature). Paper: Zhang et al. 2026. **NOT the per-NPC belief state** (that's `katgpt-sense::ReconstructionState::belief`). | Maps any "attention layer" / "linear attention" / "recurrent state" paper to Transformer-scale ops. **Do not confuse with per-NPC 8-dim belief** — different layer, different repo, different math. |
| `riir-ai/crates/riir-engine/src/cgsp_runtime/` | Curiosity-guided self-play, latent prediction SSL, MCTS collapse bridge | Runtime curiosity/exploration — maps any "self-learn" / "entropy-driven" / "collapse recovery" paper |
| `riir-neuron-db/src/` | `shard.rs` (NeuronShard Pod, `style_weights[64]`, dendritic branch), `freeze.rs` (`MerkleFrozenEnvelope`), `consolidation.rs` (Raven/δ-Mem), `gateway.rs` (AnyRAG escalation), `vibe.rs` (KG triple arch agent), `merkle.rs` (generic MerkleTree/Proof), `mape_k.rs` (self-healing loop), `spectral_flatness.rs` (lottery-ticket init), `shard_compactor.rs` | **Frozen latent-state storage + integrity + retrieval** — the persistence substrate for any "snapshot" / "integrity envelope" / "memory consolidation" / "external knowledge escalation" / "KG triple emission" paper. Maps any "memory" / "replay buffer" / "experience replay" / "spectral init" / "Merkle commitment" paper. |
| `riir-chain/src/encoding/latcal*.rs` + `latcal_fixed.rs` | Lattice Calculus: 2×2 matrix arithmetic obfuscation, fixed-point bridge, spectral fixed-point, batch determinant validation, DeFi programs | **The sync-boundary bridge** — deterministic, committed, raw-numeric. Maps any "fixed-point" / "deterministic commitment" / "raw↔latent bridge" / "arithmetic obfuscation" paper. LatCal is how latent ops become chain-committed raw values. |
| `katgpt-rs/crates/katgpt-dec/src/` | `operators.rs` (d=`exterior_derivative`, δ=`codifferential`, Δ=`hodge_laplacian`), `hodge.rs` (`hodge_decompose` exact/coexact/harmonic, `betti_numbers`, `harmonic_projector`), `flow.rs` (`DecFlowField` exact/coexact/harmonic channels), `stokes_calculus.rs` (Plan 314 wrappers: `boundary_flux_mass`, `belief_mass_divergence`, `line_integral`) — **shipped Plan 251, Research 219**. Re-exported as `katgpt_core::dec::*` via `pub use katgpt_dec as dec;` behind `dec_operators` feature. Typed game cochains (Safety/Threat/Occupancy) live separately in `katgpt-core/src/multi_agent_path/`. | **The Generalized Stokes' Theorem substrate** — `d∘d=0` enforced by construction (tests verify `curl(grad)=0`, `div(curl)=0`). Maps any "divergence" / "boundary flux" / "line integral" / "curl" / "Hodge decomposition" / "Fokker-Planck" / "mass conservation" / "manifold geometry" / "exterior calculus" / "Stokes theorem" paper. **Curse-of-dimensionality caveat: boundary-vs-volume wins only for d ≤ 3 (game maps, belief regions, KG embeddings) — NOT high-dim shards.** |

**Adapter routing, KV compression, and speculative decode are GOAT-tier framings. Latent-to-latent operations on belief/functor/neuron-shard/LatCal state are Super-GOAT-tier framings. Attempt the Super-GOAT framing first.** Defaulting to adapter routing when a latent-space reframing is stronger is the primary failure mode this protocol prevents.

## Redirect to riir-train (do NOT distill here)

**MANDATORY pre-check:** before redirecting ANY mechanism to riir-train, exhaust the modelless unblock paths in §3.5 below — **starting with Path 0 (training-target decomposition)**. A mechanism that *looks* training-only may be modelless-validable because its training-target MATH decomposes into already-shipped primitives (dllm interpolant + Latent Field Steering reward-gradient + freeze/thaw replay buffer + induced CWM direction mining). The Flow Sampling lesson (arxiv 2605.03984) is the canonical case: "trains a drift network u_θ via backprop" was initially → riir-train; the conditional drift formula actually decomposes into modelless primitives we already ship. Only redirect if §3.5's decision protocol (Path 0 + Paths 1–3) returns "genuine riir-train dependency".

If a paper is genuinely training-only (after §3.5 Path 0 + Paths 1–3 check) → **DO NOT lazily one-line redirect.** Run §3.5 Path 0.5 (training-cost-weighted re-evaluation). **As of 2026-08-06 we are near the limit of modelless gains; the model-based track is now actively pursued.** If the training insight is applicable to our model-based track (any pipeline in `riir-train/` — Plan 318 SFT/LoRA, Plan 059 GRPO/DPO, Plan 066 HLA distillation, Plan 501–505 trajectories; OR adapter training that feeds freeze/thaw via `LoraPair`), **create a Plan in `riir-train/.plans/`** (named recipe + GPU-hours estimate + GOAT gate comparing trained-vs-modelless baseline). Only do the one-line "→ riir-train" redirect if the training method is **genuinely out of scope** for our stack (e.g., image-specific DiT architecture we'll never train, medical imaging we don't do) — and justify why explicitly. The lazy "note → riir-train and stop" default is RETIRED for applicable training papers.

**By topic:**
- LoRA / OFT / SPEFT / IA3 / QLoRA / ManifoldE / BAKE / GPart / MSA / Dendritic and all adapter-**training** methods.
- Training optimizers (Muon, Adam variants, symmetry-compatible optimizers).
- Training loss functions, curricula, distillation recipes.
- Quantization-aware **training** (quantization-aware **inference** stays here).
- DPO / GRPO / SFT / RL **training** pipelines (runtime GRPO self-play stays in `riir-ai` — it updates latent state, not weights).
- Anything that requires backpropagation through base weights.

**By user-request phrasing (these mean "→ riir-train"):**
- "Train a LoRA adapter to do X"
- "Fine-tune with method Y"
- "Optimizer Z improves convergence"
- "Distillation recipe from teacher to student"
- "Quantization-aware training" (but "quantization-aware inference" stays here)
- "DPO/GRPO/SFT/RL training pipeline" (but runtime GRPO self-play stays in riir-ai)

## Distillation targets (7-repo strategy — research primarily lands in the 5 core distillation targets; SDK + armageddon are downstream consumers)

| Repo | Role | What lands here |
|------|------|-----------------|
| `katgpt-rs` (public, MIT) | Engine — modelless inference framework | Generic primitives: ConstraintPruner traits, bandits, DDTree, speculative decode, sparse attention kernels. **No game IP, no chain IP, no neuron-shard IP.** |
| `riir-ai` (private) | Game product — freeze/thaw runtime, self-learn, game systems | Runtime IP: `LoRAWeightVersion`, `LoRAHotSwap`, `dispatch_lora_merge`, `TrainingProvider` trait, routing, game systems. |
| `riir-chain` (private) | Neuro-symbolic chain transport — LatCal, chaind | Chain IP: LatCal encoding/bridges, split-key ledger, chain economics, Solana-parity features, asset lifecycle / forensic, `riir-chaind` daemon, validator SDK bridges, `catchup/` (Turso/libSQL, quorum), `DataTier` / `DATA_TIERS` / `build_tier_root` / `build_block_root`. **Re-exports `riir-neuron-db` via `neuron_db` feature, but the shard source of truth is `riir-neuron-db/`.** |
| `riir-neuron-db` (private) | Neuron-shard leaf crate — shards, freeze, consolidation, retrieval | Shard IP: `NeuronShard` Pod layout + `style_weights[64]` + dendritic branch, `ShardIndex` lock-free papaya, generic `MerkleTree`/`MerkleProof`, `MerkleFrozenEnvelope`, MAPE-K self-healing, Raven/δ-Mem consolidation, AnyRAG escalation gateway, vibe KG triple templates + arch agent, spectral lottery-ticket init, `ShardCompactor`. **No chain dependency — usable standalone.** |
| `riir-train` (private) | Training research vault | **Only if the paper's value is its training method.** **As of 2026-08-06: training-efficiency papers applicable to our model-based track are ACTIVELY pursued (not lazily redirected).** Create a Plan in `riir-train/.plans/` per §3.5 Path 0.5 with recipe + GPU-hours estimate + GOAT gate. Only genuinely out-of-scope training gets the one-line redirect. |
| `riir-game-sdk` (private) | Game-vocabulary facade + dev-tool workspace | **Downstream consumer** — typically not a distillation target. Research lands here only when the paper's insight is about SDK-facing vocabulary, builder patterns, or backend abstractions (rare). The vocabulary substrate itself lives in `riir-games-shared` (in riir-ai workspace). |
| `riir-armageddon` (private) | Arena/game-product domain types | **Downstream consumer** — typically not a distillation target. Research lands here only for product-domain type design (rare). |

Distill into:
- **Modelless** → `katgpt-rs/.research/` + `katgpt-rs/.plans/` + `katgpt-rs/src/` (or `katgpt-rs/crates/katgpt-core/`)
- **Runtime/game** → `riir-ai/.research/` + `riir-ai/.plans/` + `riir-ai/crates/`
- **Chain / LatCal / sync-bridge / commitment / quorum / catchup** → `riir-chain/.research/` (create if missing) + `riir-chain/.plans/` + `riir-chain/src/` (or `riir-chain/crates/`)
- **Neuron shards / freeze envelope / consolidation / AnyRAG / vibe KG / Merkle tree / spectral init / shard compaction** → `riir-neuron-db/.research/` (create if missing) + `riir-neuron-db/.plans/` + `riir-neuron-db/src/`
- **Training-only** → run §3.5 Path 0.5: if applicable to model-based track → create a Plan in `riir-train/.plans/` (NOT a lazy redirect, as of 2026-08-06). If genuinely out of scope → note the redirect with explicit justification.

## Workflow

### 0. Read & classify the paper

Fetch via `https://r.jina.ai/https://arxiv.org/pdf/{ID}` (per AGENTS.md). Ask: *is the value in the training loop itself (optimizer / loss / schedule / RL algorithm), or in the math the training computes (closed-form drift, conditional score, Riemannian correction, steering formula)?* If optimizer/loss/schedule → riir-train. **If the math → run §3.5 Path 0 (training-target decomposition) FIRST**: decompose the math into components and grep for the modelless analog of EACH before redirecting. Only if no component has a modelless analog → riir-train.

**Five "not automatically PASS/redirect" lessons — always ask the key question before classifying:**

| Paper class | Lesson | Key question before PASS/redirect |
|---|---|---|
| **Training-math** (Flow Sampling, arxiv 2605.03984) | "train via backprop" decomposed into shipped primitives (dllm + Latent Field Steering Plan 309 + freeze/thaw) | Does the math decompose into modelless components? (§3.5 Path 0). **PTRM caution (R049):** a shipped modelless analog is necessary but NOT sufficient — there must also be a use case. |
| **Hardware/accelerator/NMP/PIM/ASIC** (R418 StreamDQ arxiv 2607.08993) | "hardware-only" was wrong — value was the technique (LUT INT→FP, shared ALU, sideband-tag dispatch), substrate-independent. Shipped 2.3×. | What is the *technique* stripped of the hardware substrate? Grep `simd_*`, `ternary`, `Plasma`, `Q4_K`, `LUT`, `from_bits` for the software analog. (Research 110 Ciot proves we simulate hardware in software.) |
| **Database/systems/storage** (R300 Trellis arxiv 2606.29823) | "no database engine" was wrong — `riir-neuron-db` IS a database (Pod + ShardIndex + Merkle + MAPE-K + Raven/δ-Mem + vibe KG). Value was the access pattern. | What is the *access pattern*? Grep `riir-neuron-db/src/` + `episode_buffer.rs` + `cgsp_runtime/persistence.rs`. **Conflating "no fused query planner" with "no database" is the false-PASS root cause.** |
| **Pure math/combinatorics/game theory** (Lonely Runner, arxiv 0710.4495, R470) | "non-constructive proof, no algorithm" was wrong — value was guaranteed-peak property on a per-entity scalar (`phase_separation`), modelless, sync-safe. | What per-entity scalar does this theorem BOUND, and can it drive a game-system behavior? See §1 step 4 for the MANDATORY game-context reframe. |
| **System/kernel/scheduler/OS** (io_uring, DPDK, RDMA) | Substrate (Linux kernel) is not the value; the technique (lock-free queue, batching, zero-copy, prefetch) often is. | What is the *technique* stripped of the OS substrate? |

**Hard rule — substrate ≠ value:** Before PASS-ing on ANY paper whose abstract contains hardware vocabulary (HBM, ASIC, RTL, NMP, PE, DQB, tensor-core, TPU, FPGA, accelerator, near-memory, in-memory compute, PIM): (1) identify the technique stripped of the substrate, (2) grep for the software-SIMD analog, (3) PASS ONLY if both checks confirm no analog exists.

### 1. Distill fundamentally — fuse, don't just direct-map

Don't direct-map the paper to our code. Find the transferable primitive: the geometric, spectral, or information-theoretic insight that works without the paper's training setup. **Then look for fusion opportunities**: cross-pollinate this paper's insight with existing `.research/` notes, `.plans/`, and shipped primitives to synthesize a *novel* combination. The highest-value Super-GOATs in freeze/thaw runtime and self-learn/adaptive CoT almost always come from **fusing** 2–3 papers, not from a single-paper direct mapping.

**Fusion examples that shipped (two patterns — cross-repo + multi-primitive):**
- **Cross-repo:** Gemini Fourier × LatCal → `katgpt-rs/.research/212_*` → `katgpt-rs/.plans/242_*` (a `katgpt-rs` modelless primitive fused with a `riir-chain` commitment bridge)
- **Multi-primitive:** Collapse-aware × bandit coverage × sigmoid margin → `katgpt-rs/.plans/212_*` × `157_*` × `061_*` (three inference primitives fused into one collapse-recovery gate)

**Fusion protocol:**
1. **MANDATORY — grep ALL SEVEN repos in this session, BOTH layers (notes AND code). Do NOT stop after the first repo or the first layer. Do NOT wait for the user to prompt you repo-by-repo — run ALL greps in one pass, preferably in parallel via subagents.** (The Research 411 lesson, 2026-08-09: the agent only grepped the home repo (riir-train/.research) on the initial pass, then waited for the user to manually prompt "grep katgpt-rs/.research", "grep riir-ai/.research" — three separate user turns. ALL THREE greps were mandatory by this instruction and should have run unprompted in the initial pass. The riir-ai grep later found Research 328 — the transformer-layer MoE substrate — which materially changed the verdict.) Run keyword / paper-title / author / primitive-name grep across:
   - `katgpt-rs/.research/` + `katgpt-rs/.plans/` (intent — what we planned)
   - `riir-ai/.research/` + `riir-ai/.plans/` (intent — runtime/game)
   - `riir-chain/.research/` + `riir-chain/.plans/` (intent — current chain research; `.research/` may need creating on first use)
   - `riir-neuron-db/.research/` + `riir-neuron-db/.plans/` (intent — current shard research; `.research/` may need creating on first use)
   - `riir-ai/.docs/` — the consolidated selling-point book (`pillars/`, `04_supergoat_candidates/`, `reasoning/`, `self_learn_npcs/`, `neuro_symbolic_chain/`, ...). These are not academic distillation; they are the moat/selling-point framing. Grep them alongside `.research/` so you do not claim novelty over a pillar that already ships.
   - `katgpt-rs/src/` + `katgpt-rs/crates/` (shipped primitives — what actually exists)
   - `riir-ai/crates/` (shipped runtime)
   - `riir-chain/src/` + `riir-chain/crates/` (shipped chain — LatCal, encoding, economics, forensic, catchup, etc.)
   - `riir-neuron-db/src/` (shipped shards — `shard.rs`, `freeze.rs`, `consolidation.rs`, `gateway.rs`, `vibe.rs`, `merkle.rs`, `mape_k.rs`, `spectral_flatness.rs`, `shard_compactor.rs`)
   - **Super-GOAT factory modules** (from §Primary focus) — `list_directory` these explicitly even if the paper looks pure-training: `katgpt-rs/crates/katgpt-core/src/sense/`, `riir-ai/crates/riir-engine/src/latent_functor/`, `riir-ai/crates/riir-engine/src/hla/`, `riir-ai/crates/riir-engine/src/cgsp_runtime/`, `riir-neuron-db/src/` (shards/freeze/consolidation/AnyRAG/vibe/merkle), `riir-chain/src/encoding/latcal*.rs`, `katgpt-rs/crates/katgpt-dec/src/` (Stokes/exterior-derivative/Hodge — maps any divergence/boundary/line-integral/Fokker-Planck/manifold-geometry paper)

   (riir-train is NOT excluded — training-efficiency papers applicable to our model-based track get a Plan in `riir-train/.plans/` per §3.5 Path 0.5. Grep `riir-train/.research/` + `riir-train/.plans/` as part of the fusion search for training-method papers. The old "deliberately excluded" framing is retired 2026-08-06 — the modelless track is near its limit, the model-based track is actively pursued.)

   Two layers, seven repos (5 primary distillation targets + 2 downstream consumers). The closest cousin is frequently in the OTHER repo (e.g., a `katgpt-rs` modelless primitive fused with a `riir-chain` LatCal commitment bridge — see Gemini Fourier × LatCal; or a `riir-neuron-db` freeze envelope fused with a `riir-ai` runtime adapter hot-swap) OR in the CODE not the notes. **Notes describe intent; code describes what shipped.** A mechanism can ship without a research note — e.g., belief's `evolve_belief` (`katgpt-rs/crates/katgpt-core/src/sense/reconstruction.rs`) is a per-NPC recurrent belief-state kernel with no `.research/` note framing it as such; a notes-only grep misses it and produces a false Super-GOAT claim (verdict then has to be revised down). If you only grep `katgpt-rs/.research/`, you will miss both axes and produce a duplicate, weaker note, or an overclaimed verdict.

2. **MANDATORY — vocabulary translation before grepping.** Papers and our codebase use different words for the same mechanism. Before any grep, list the paper's 3–5 key mechanism terms, then for EACH, brainstorm ≥2 codebase-equivalent terms by asking: "if we shipped this, what would we call it?" Then grep BOTH sets.

   **`read_file` `vocab.md`** (this skill's sibling file) for the 6 standing vocabulary tables + unified decision rule + worked examples. The tables cover: (1) latent-state, (2) per-NPC runtime/freeze-thaw/personality, (3) compute-unit translation (for agent/LLM papers — R368 lesson), (4) substrate-translation (for hardware/accelerator/NMP/PIM papers — R418 lesson), (5) database-substrate translation (R300 lesson), (6) the unified substrate-as-instantiation-vs-mechanism-as-value decision rule. Always include table (1) latent-state even for non-latent papers; include the domain-specific tables (2–5) when the paper touches that domain.

3. **MANDATORY — latent-space reframing before verdict.** Before any verdict, re-cast the paper's core mechanism as a latent-to-latent operation on the codebase's latent-state kernels (the seven Super-GOAT factory modules above). Ask explicitly: "How does this mechanism look when operating on (a) per-NPC belief latent state, (b) `latent_functor/` operations, (c) `cgsp_runtime/` curiosity signals, (d) LatCal fixed-point commitment (in `riir-chain/src/encoding/`), (e) `NeuronShard` style_weights / dendritic branch / `MerkleFrozenEnvelope` / Raven consolidation / AnyRAG escalation (in `riir-neuron-db/src/`), (f) DEC Stokes-calculus operators (`katgpt-rs/crates/katgpt-dec/src/` — `exterior_derivative` d, `codifferential` δ, `hodge_decompose`, `DecFlowField` exact/coexact/harmonic)?" If your fusion idea only touches adapter routing / KV compression / speculative decode without a latent-state reframing, you are likely in GOAT territory and have probably missed the Super-GOAT angle. If you find yourself reaching for an adapter-routing framing, treat it as a symptom that the stronger latent-functor / belief / neuron-shard / LatCal reframing is still unfound — adapter routing is the fallback, never the primary Super-GOAT framing. The latent reframing is mandatory even for papers that look pure-training/architecture — most have a latent subspace / stage-gating / persistence / memory-consolidation / manifold-geometry angle that lands in belief/functor/neuron-shard/DEC.

4. **MANDATORY — game-context reframe before verdict (the Lonely Runner lesson, alongside the latent-space reframe above, especially when the latent-space reframe returns no hits).** The latent-space reframe (step 3) asks "how does this look as an operation on belief / functor / shard / DEC state?". The game-context reframe asks a DIFFERENT question: **"how does this mechanism manifest as a per-NPC behavior signal / crowd pattern / selling point in the MMORPG game context?"** These are complementary, not redundant — a paper can have zero latent-space analog but a strong game-context application (the canonical Lonely Runner failure: torus coverage didn't map to latent ops → PASS; per-NPC guaranteed-solo-moments behavior signal → Super-GOAT). Before any verdict, explicitly ask:
   - If the paper describes a **guarantee** (existence theorem, lower bound, coverage property, fairness property) → what per-entity scalar does it bound, and can that scalar drive a behavior (salience emit cadence, curiosity trigger, consolidation window, sleep schedule, rest cycle)?
   - If the paper describes a **combinatorial structure** (graph coloring, scheduling, packing, covering, chromatic number) → does that structure appear in NPC routines, market cycles, quest scheduling, spatial coordination, or resource allocation?
   - If the paper describes a **number-theoretic property** (p-adic valuations, Diophantine approximation, lattice coverage, modular arithmetic) → does that property underwrite a fairness / diversity / coverage guarantee on a game signal?
   - If the paper describes a **geometric / topological property** (manifold coverage, torus orbits, covering radius, fundamental domain) → does that property map to NPC phase scheduling, spatial spread, or fog-of-war coverage?

   **If the latent-space reframe returned "no analog" AND you have NOT done the game-context reframe, you are NOT ready to PASS.** Run step 4 first. The game-context reframe is mandatory even for papers that look pure-math / pure-combinatorics / pure-game-theory — these are the papers MOST LIKELY to have a behavior-signal application that the latent-space reframe misses, because behavior signals live in the game-runtime layer (riir-ai pillars: Salience, Sleep-Time, KARC, feeling brain, motivation brain, swarm coordination), NOT in the latent-state kernel layer. Canonical case: Research 470 (Lonely Runner Conjecture → `phase_separation` primitive → 5-pillar fusion → Super-GOAT).

5. **Zero grep hits ≠ novelty.** If your paper-vocabulary grep AND your codebase-vocabulary grep BOTH return zero hits, that is evidence of one of three things, in order of likelihood: (a) you are still using the wrong vocabulary — try a third semantic angle (e.g., grep for the *output behavior* like "swap when X" instead of the *mechanism name* like "tightness monitor"); (b) the mechanism is genuinely not shipped; (c) the mechanism is novel. Do NOT jump to (c). Default to (a): re-grep with at least one more semantic alternative before claiming "no prior art".
6. After finding the transferable primitive of *this* paper, list the 2–3 closest existing notes/plans **across all seven repos** and ask: "what novel combination of this paper + note A + note B produces a capability none of them has alone?" Write that combination into the research note's §Distillation as a **Fusion** subsection, even if you don't plan it yet.
7. Verdict by the commercial strategy tiers (see §Cross-references for the strategy doc): **Super-GOAT** > GOAT > Gain > Pass (see §Verdict tiers below). **A fusion that produces a new capability class is a strong Super-GOAT candidate — check the novelty gate (§1.5).**
8. Create research `.md` at the right repo (see table above).

**File naming:** `{NNN}_{Short_Title_with_Underscores}.md` where NNN is the next free number (zero-padded to 3 digits). Numbers are monotonic and never reused.

**Research note format:** `read_file` `templates.md` (this skill's sibling file) for the canonical research note + plan format templates, the verdict tier table, and the UQ-bearing "Report the Floor" GOAT gate rule. Canonical example: `katgpt-rs/.research/238_LoRA_Muon_Spectral_Low_Rank_Manifold.md`.

### 1.5. Novelty gate — is this Super-GOAT?

Before planning, score novelty. Ask all four:

1. **No prior art?** Grep `.research/` + `.plans/` across all repos AND grep the shipped code (`katgpt-rs/src/`, `katgpt-rs/crates/`, `riir-ai/crates/`, `riir-chain/src/`, `riir-chain/crates/`, `riir-neuron-db/src/`) for the primitive name and mechanism keywords. **You MUST grep BOTH paper vocabulary AND codebase-vocabulary alternatives (see §Workflow fusion protocol step 2 — vocabulary translation).** **Notes describe intent; code describes what shipped.** A mechanism can ship under either of two failure modes:
   - **No notes framing at all** — canonical example: belief's `evolve_belief` (`katgpt-rs/crates/katgpt-core/src/sense/reconstruction.rs`) is a per-NPC recurrent belief-state kernel with no `.research/` note framing it as such; missing it has historically caused false Super-GOAT claims.
   - **Notes framing uses different vocabulary than the paper** — canonical example: DiPOD's "interleave self-distillation when ELBO drifts" is shipped as `riir-ai/crates/riir-engine/src/latent_functor/reestimation.rs` "coherence-driven re-estimation scheduler when coherence < tau_reest". The note DOES frame the mechanism, but using codebase vocabulary, so a paper-vocabulary grep misses it on BOTH notes AND code layers. This is strictly worse than the `evolve_belief` failure: even a diligent notes grep fails. **Vocabulary translation (fusion protocol step 2) is the only defense.**
   If the code already covers the mechanism → not novel, Gain at best. **This three-layer check (notes + code + vocabulary translation) is mandatory — notes-only is the #1 cause of false Super-GOAT claims; paper-vocabulary-only is the #2 cause; skipping the seven Super-GOAT factory modules is the #3 cause.**

   **Grep returns candidates; READING the candidates is mandatory.** A grep hit is a lead to follow, not a prior-art confirmation. When a grep hit's filename or first-line summary touches the candidate's selling-point space (per-NPC, memory, personality, swap, freeze/thaw, evaluator, critic, curiosity, test-time scaling, sleep-time, sub-goal), `read_file` the hit's TL;DR + §1 (selling point) BEFORE claiming novelty. Grepping `riir-ai/.research/`, seeing a filename match, and moving on is the failure mode: the guide frames the mechanism under different vocabulary, so the filename looks unrelated even though the content is exact prior art. **When the candidate selling point touches per-NPC + memory + personality + swap, the `riir-ai/.research/` corpus is saturated — grep it for `Per_NPC|Committed|Cognitive_Branch|Sub_Goal|Karc|CLR|Sleep_Time|Gain_Cost|Personality|Curiosity|Mind_Reading` and READ every hit's TL;DR before claiming novelty. Assume covered until proven otherwise.**
2. **New class of behavior?** Not better numbers, but something no incumbent can do (a new capability, not an optimization). **Requires the game-context reframe (§1 step 4) to detect** — "guaranteed individuality moments" (Lonely Runner, Research 470) only appears as a new behavior class AFTER the game-context reframe, not from the latent-space reframe alone. If you reached Q2 without running step 4, go back.
3. **Product selling point?** Can you finish the sentence: "Our NPCs/systems do X that no competitor can"? If you can't → Gain.
4. **Force multiplier?** Connects to ≥2 existing pillars/systems (check connection map in `.research/`). Solo novelty without integration = GOAT, not Super-GOAT.

**If YES to all 4 → verdict = Super-GOAT.** Mandatory outputs:
1. **Open primitive** → `katgpt-rs` (generic math, no game semantics).
2. **Architectural GUIDE** → the private selling-point doc. **Pick the repo by where the selling point lives**: `riir-ai/.research/NNN_*.md` for game-runtime / belief / functor / self-learn selling points; `riir-chain/.research/NNN_*.md` for chain / LatCal / commitment / quorum / catchup / sync-bridge selling points (create folder on first use); `riir-neuron-db/.research/NNN_*.md` for shard / freeze envelope / consolidation / AnyRAG / vibe KG / Merkle tree / spectral init / shard compaction selling points (create folder on first use). If the selling point spans multiple repos (e.g., latent ops that cross the chain sync boundary via a shard commitment), create the primary guide in the repo that owns the boundary being crossed, and cross-reference from the others. The guide MUST include:
   - TL;DR with commercial value (the selling point in one sentence)
   - Distilled primitive (how the mechanism works modellessly)
   - Connection map (which existing systems it multiplies)
   - Latent vs raw boundary (what crosses sync, what stays local)
   - What stays private vs open
   - Validation protocol (how to prove it's Super-GOAT, not just hype)
   - Implementation priority table (P0–P3)
3. **Plan(s)** → `katgpt-rs/.plans/` (open) and/or `riir-ai/.plans/` (private runtime) and/or `riir-chain/.plans/` (private chain) and/or `riir-neuron-db/.plans/` (private shards).

**If NO to any → proceed to GOAT/Gain verdict.** Plan only, no guide.

> **Rule:** Super-GOAT ideas are the private IP moat. The open primitive is the adoption hook; the riir-ai/riir-chain/riir-neuron-db guide is the selling point. Never ship the guide publicly. Never skip the guide for a Super-GOAT — that's losing the knowledge.
>
> **No "candidate" escape hatch.** If you write "all 4 YES", "passes the novelty gate", or "Super-GOAT candidate" anywhere in a note (main verdict OR a fusion subsection), the mandatory outputs above apply **in this same session** — open primitive in katgpt-rs, **private guide (riir-ai OR riir-chain OR riir-neuron-db, by selling-point domain) created now**, plans as needed. The guide *contains* the validation protocol (G1–Gn gate), so you create it **before** running the gate, not after. Deferring the guide "until validation passes" inverts the order and silently drops the moat doc — this is the #1 way selling points leak into the public repo.
>
> If you are NOT confident enough to commit all 4 YES right now, **do not write "Super-GOAT candidate"**. Write "fusion idea — novelty TBD, needs Q1–Q4 check before verdict" and create an issue in `.issues/` to track the follow-up. "Candidate" is not a deferred-commitment escape hatch — it either triggers the guide now, or it gets downgraded to an issue.

### 1.55. PASS vs Gain — no middle tier

**The rule is simple: PASS = no new research/plan files. Gain = files.**

Before any verdict, scan the paper for actionable improvements to our stack (config changes, PoC tasks, unblockers, unmitigated failure modes). Then:

- **If the mechanism ships AND there are actionable improvements** → verdict is **Gain**, not Pass. Create `.issues/` entries (per AGENTS.md: "Create issue at .issues for poc, proof, optimization or refactor task, do not create plan") and/or a plan behind a feature flag.
- **If the mechanism ships AND there are no actionable improvements** → verdict is **Pass**. No new research/plan files. Report verdict + one-line reason + closest shipped cousin in conversation only.
- **If the mechanism does not ship AND it's modelless** → verdict is **Gain** or higher.
- **If the mechanism does not ship AND it's training-only** → run §3.5 Path 0.5. **Do NOT lazily one-line redirect (as of 2026-08-06).** If the training insight is applicable to our model-based track (any pipeline in `riir-train/` — Plan 318 SFT/LoRA, Plan 059 GRPO/DPO, Plan 066 HLA distillation, Plan 501–505 trajectories) → create a Plan in `riir-train/.plans/` (this is Gain for the model-based track, NOT Pass — the modelless track is near its limit, the model-based track is actively pursued). Only one-line redirect if genuinely out of scope for our stack (explicit justification required).

There is no "PASS-with-gains". There is no middle tier. If the paper produces something actionable, it's a Gain. If it doesn't, it's a Pass.

#### 1.55.1. PASS cross-reference rule (MANDATORY) — future-proofing grep

**PASS verdicts must still update the 1–3 closest shipped cousin `.research/` notes with a one-line `PASS-Redirects:` reference.** No new file — the existing note gains a line. This prevents **paper-number invisibility** (the #4 false-novelty failure mode): a future session greps for `arxiv:XXXX.XXXXX` or the paper title, finds nothing, and re-distills from scratch.

**Format** (add to cousin note header near `Related Research:`):
```
> **PASS-Redirects (synthesis):** <Author_or_Short_Title> [arXiv:XXXX.XXXXX "<Full Title>"] — <one-line reason: which shipped primitive covers it, why training-only, etc.>. <If split-stage: which stage → which repo.>
```
The line MUST include both the arxiv ID AND the full title (so both `grep arxiv:ID` AND `grep "Title"` hit). If no shipped cousin exists (genuinely out of scope), add to the closest **topic-adjacent** note.

**Actionable = Gain, not Pass.** Actionable: (a) paper data **contradicts** a current config default (grep-confirmed); (b) exposes a failure mode with **no existing mitigation**; (c) unblocks a known deferred task. NOT actionable: "validates our design" / "theoretical lens" / "could inform a future config". If unsure → not actionable → Pass.

#### 1.55.2. Documented-limitation reverse-grep (MANDATORY before PASS)

**The BTM lesson (arXiv:2608.01692):** a paper PASS'd as "we don't ship generative image models" was actually Gain — its core equation (`∇·(νb)=μ₀−μ₁`) was the EXACT primitive our shipped CCE Moderator (Plan 295) documented as missing. §1.55's actionable scan was one-directional (paper → codebase); it must be **bidirectional**.

**Before any PASS verdict, reverse-grep the codebase for documented gaps the paper could fill:**
1. Grep `.docs/` for `Limitation|deferred|follow.up|TODO|FIXME|gap|pending|not.yet` near the paper's domain.
2. Grep `.benchmarks/` for `Caveat|deferred|artifact|known.*limitation|pending`.
3. Grep shipped `.rs` comments for `TODO|FIXME|deferred|follow.up|limitation` near the paper's mechanism vocabulary.
4. If ANY hit maps to the paper's mechanism → **Gain**, not Pass.

**Compact heuristic:** before PASS, ask: *"Is there any documented limitation, deferred task, or known artifact that this paper could fix?"* If you can't answer "no, I checked" with evidence → don't PASS.

**Third defense (training papers — GDN-blog lesson 2026-08-07):** both defenses above can return clean and still produce a false PASS if the agent narrows "model-based track" to "Plan 318 SFT". Before PASS-ing any training paper, `read_file riir-train/.docs/02_pipelines/training_data_pipeline.md` + `list_directory riir-train/crates/riir-train-gpu/src/` and ask: "does a training pipeline for this paper's domain already exist in `riir-train/`?" (RL, distillation, linear attention, trajectory collection). If yes → Gain, NOT Pass. **All three defenses mandatory before PASS on training papers.**

### 1.6. MOAT gate per domain

The global verdict tiers (Super-GOAT / GOAT / Gain / Pass) measure *how strong* a contribution is. The **MOAT gate per domain** measures *whether a contribution strengthens THIS repo's moat*. A primitive can be a clean GOAT win yet contribute nothing to the moat if it lands outside the repo's pillar scope, or if a stronger latent reframing was missed. **Check the domain MOAT gate at verdict time — a mismatch means reroute to the correct repo.**

| Domain | MOAT contribution bar | In scope | Out of scope (reroute) |
|--------|----------------------|----------|----------------------|
| **`katgpt-rs`** (public engine) | **Paper-derived fundamental / principle / base-foundation primitive** that passes GOAT or Gain via fusion, with **promote/demote tracked per stack**. Aim: research-grade primitives the adoption funnel depends on. Each primitive ships behind a feature flag; the GOAT gate decides promote-to-default vs demote-loser per stack. | Transformer stack (layers, attention, KV cache, sampling, sparse / quant-aware **inference**, speculative decode, DDTree, MCTS, bandits, ConstraintPruners); **2D toy benchmark games** (bomber/go/monopoly/fft-arena) + their generic MCTS/bandit/CCE wiring; DEC/Stokes substrate; belief kernel; sigmoid mechanics. | Product game wiring (→ riir-ai); chain commitment (→ riir-chain); shard internals (→ riir-neuron-db); trained weights (→ riir-train). |
| **`riir-ai`** (private runtime) | **Pillar-level or Super-GOAT**: fusion-GOAT / fusion-Gain that connects to ≥2 pillars, OR a new pillar candidate (sloppy-test winner). | **Adaptive / self-learn NPCs**, **reasoning pack** (P8), **MMORPG-scale** (20Hz tick, fog-of-war, zone attention, crowd MCGS), **3D game wiring**, freeze/thaw runtime, latent-to-latent ops on belief/functor/cgsp state. | Generic transformer mechanics (→ katgpt-rs); chain transport (→ riir-chain); shard storage (→ riir-neuron-db); training methods (→ riir-train). |
| **`riir-chain`** (private chain) | **Pillar-level or Super-GOAT**: pillar 3 (riir-chain) amplifier, OR sync-boundary bridge novelty. | LatCal commitment, quorum/catchup, chain economics, asset lifecycle / forensic, DeFi programs, `riir-chaind`, the raw↔latent sync-boundary bridge. | Generic fixed-point math without commitment semantics (→ katgpt-rs); shard internals (→ riir-neuron-db). |
| **`riir-neuron-db`** (private shards) | **Pillar-level or Super-GOAT**: pillar 2 (riir-neuron-db) amplifier, OR shard/freeze/consolidation novelty. | `NeuronShard` layout, freeze/thaw envelope, Raven/δ-Mem consolidation, AnyRAG escalation, vibe KG triples, Merkle integrity, spectral init, shard compaction, dendritic branch. | Chain commitment of shards (→ riir-chain); runtime adapter swap (→ riir-ai). |
| **`riir-train`** (private training) | **Active moat (as of 2026-08-06)**: training-method implementations + configs + trained weight assets (GPU-hours). The modelless track is near its limit; training efficiency is now actively pursued. Training-efficiency papers applicable to our model-based track MUST get a Plan in `riir-train/.plans/` per §3.5 Path 0.5 — NOT a lazy one-line redirect. | Adapter training, optimizers, loss functions, quant-aware **training**, DPO/GRPO/SFT pipelines, trained weight assets. | Inference-time / runtime / latent ops (→ katgpt-rs or riir-ai). |

**Pillar reference (riir-* repos):** the 9 sloppy-test winners live in `riir-ai/.docs/03_pillars/README.md` — **`read_file` `03_pillars/README.md` + `04_supergoat_candidates/README.md` before any "does this become a pillar?" MOAT verdict.** The 4-layer architecture (Foundation → AI → Emergent → Delivery, strict downward dependency) and the sloppy test (*if it doesn't exist, the system goes structurally sloppy — not slower, broken*) define what a pillar-level contribution means. The 9 pillars: (1) Egg/Shell + Bridge, (2) riir-neuron-db, (3) riir-chain, (4) Fourier Spatial AI, (5) WASM Validators, (6) NPC Dialog Engine, (7) Frame-Sampling Bridge, (8) Reasoning Pack, (9) Asset Vessel.

**MOAT verdict (per contribution):**
- **Strengthens moat** (in-scope pillar-level / Super-GOAT / fusion-GOAT connecting ≥2 pillars) → promote aggressively; if Super-GOAT, capture the private guide now (§1.5).
- **Neutral GOAT/Gain** (in-scope but not pillar-level) → ship behind feature flag, track promote/demote, do NOT overclaim moat in the note.
- **Out-of-scope** → reroute to the correct repo (7-repo discipline). A great primitive in the wrong repo dilutes the moat — e.g. a generic attention kernel merged into `riir-ai` instead of `katgpt-rs` leaks nothing privately valuable but starves the public adoption funnel.

**`katgpt-rs` promote/demote tracking (per stack):** every primitive that lands in the public engine gets a feature flag + benchmark + GOAT gate, and the verdict note MUST record the per-stack outcome — which transformer stack slot (attention / KV / sampling / speculative / pruning) and whether it promoted to default or stayed opt-in. Re-gate on feature touch. Demote the loser when a newer primitive wins the same slot. This per-stack ledger is the engine's quality contract.

### 1.7. Pre-plan cherry-pick audit (if consuming a katgpt-rs primitive)

**If your plan will consume, wire, or fuse with a katgpt-rs primitive into riir-*** — run the `goat-audit` skill before opening the plan. The audit answers two questions that prevent duplicate work:

1. **Is the primitive already wired into riir-\*?** (stall detection — default-on in katgpt-rs for ≥7 days with zero riir-\* consumer = candidate gap, OR already wired = no plan needed)
2. **Is riir-\* shipping a local duplicate of the substrate?** (DRY violation — the Issue 019 class: `riir-engine/src/transformer/mod.rs` defined its own `KVCache`/`KVSnapshot`/`PAGE_SIZE` instead of consuming `katgpt-transformer` — the dep was declared but unused. Plan 406 de-forked these.)

**When to run goat-audit:**
- The plan's target repo is riir-ai / riir-chain / riir-neuron-db AND the plan consumes a katgpt-rs feature/struct/function.
- The plan is a Super-GOAT fusion that touches a katgpt-rs primitive + a riir-\* runtime.
- Quarterly hygiene gate (re-audit after every major katgpt-rs release).

**When NOT to run goat-audit:**
- The plan is purely katgpt-rs-internal (no riir-\* consumer).
- The plan is a bug fix with no cross-repo angle.
- The plan is training-only AND genuinely out of scope for our model-based track (→ riir-train one-line redirect with justification; per §3.5 Path 0.5, applicable training papers DO get a plan in `riir-train/.plans/`).

Invoke via the `skill` tool with name `goat-audit`. The skill's three-layer grep (feature-name + struct/function-name + consumer-vs-duplicate) catches both false negatives (Issue 003's `salience_tri_gate` miss) and false positives (Issue 019's `KVCache` local-shadow duplicate flagged as wired).

### 2. If gain (or GOAT), plan it

Add plan `.md` to `katgpt-rs/.plans/` (modelless), `riir-ai/.plans/` (runtime/game), `riir-chain/.plans/` (chain / LatCal / neuron_db), and/or **`riir-train/.plans/` (training-efficiency papers that pass §3.5 Path 0.5)**. Use `## Phase N` sections with `- [ ]` per task (mark `- [x]` when done). Planning into riir-train IS allowed — and now actively encouraged — for training-efficiency papers applicable to our model-based track (the modelless track is near its limit as of 2026-08-06; the model-based track is the growth frontier; it is broader than Plan 318 — includes Plan 059 GRPO/DPO, Plan 066 HLA distillation, Plan 501–505 trajectories). The old "Never plan into riir-train" ban is RETIRED.

> Super-GOAT plans should be created AFTER the riir-ai guide. The guide is the strategy; the plan is the execution.

**Plan format:** `read_file` `templates.md` (this skill's sibling file) for the canonical plan format template + the GOAT gate rule + the UQ-bearing "Report the Floor" extension (Research 322). Canonical example: `katgpt-rs/.plans/271_attention_matching_compaction.md`.

**Compact GOAT gate rule:** every plan introducing a new technique needs a feature flag + a benchmark proving the gain before promoting to default. Demote the loser if the new technique wins the same slot. UQ-bearing primitives (probability distributions, predictive intervals, coverage guarantees) MUST additionally benchmark against the conformal-naive floor (`ConformalIntervalCalibrator<SeasonalNaiveForecaster>`, Plan 340) — if they can't beat it on CRPS / coverage / Winkler score, the GOAT gate FAILS. See `templates.md` for the full rule.

### 3. Implement to unblock

If a plan is blocked by a missing primitive, implement the minimal version. After GOAT check + proof of gain: promote to default if it wins, demote the loser.

### 3.5. Modelless unblock protocol — MANDATORY before any riir-train deferral

**Hard rule:** before deferring ANY GOAT gate, plan task, or mechanism to riir-train ("this needs training"), you MUST exhaust all modelless correction paths first. A gate that *appears* to need training may be passable modellessly via training-target decomposition, freeze/thaw, raw/lora hot-swap, or latent-space correction. Deferring to riir-train without checking is the failure mode this protocol prevents.

**Path 0 — training-target decomposition (the Flow Sampling lesson, arxiv 2605.03984):** Before checking paths 1–3, ask: "is the mechanism's value the **training-loop innovation itself** (new optimizer/loss/curriculum/RL algorithm), or the **math the training computes** (closed-form drift, conditional score, Riemannian correction, steering formula, regression target)?" If the latter, decompose the math into components and grep the codebase for the modelless analog of EACH component (see §1 step 2 standing training-math vocabulary). If ALL components have modelless analogs → the mechanism is MODELLESS-VALIDABLE as a fusion of existing primitives — no riir-train deferral needed. **Canonical: Flow Sampling** — the conditional drift formula decomposes into dllm interpolant + Latent Field Steering (Plan 309) reward-gradient + freeze/thaw replay buffer; the training loop is just one way to compute that math. Only if the math has no modelless decomposition → proceed to paths 1–3 below.

**Path 0.5 — training-cost-weighted re-evaluation (the Plan 318 amendment, adopted 2026-08-05; ELEVATED TO DEFAULT for training-efficiency papers 2026-08-06; BROADENED 2026-08-07 per the GDN-blog lesson):** **As of 2026-08-06, we are near the limit of modelless gains. Training efficiency is now actively pursued on the model-based track.** Any training-efficiency paper (optimizer, loss, schedule, recipe) that applies to our model-based track MUST get a Plan in `riir-train/.plans/`, NOT a lazy one-line redirect. The old "note → riir-train and stop" default is RETIRED for applicable training papers; it survives ONLY for genuinely out-of-scope training (e.g., image-specific architecture we'll never train).

**The model-based track is broader than Plan 318 SFT (the GDN-blog lesson, 2026-08-07).** An initial PASS verdict on a GDN train-inference mismatch blog narrowed "model-based track" to "LLM/LoRA training via Plan 318" and concluded the paper didn't apply. Wrong — `read_file riir-train/.docs/02_pipelines/training_data_pipeline.md` would have shown the track includes: Plan 318 (SFT + LoRA fine-tune) + **Plan 059 (G-Zero DPO/GRPO — real RL training)** + **Plan 066 / `distill_attention.rs` (SDPA→HLA/AHLA distillation — linear attention training)** + Plan 501–505 (trajectory collection). The paper touched TWO of those (RL + linear attention). Verdict revised PASS→Gain. **Lesson: the model-based track = ALL training pipelines in `riir-train/`, not just Plan 318. Before PASS-ing any training paper, `read_file riir-train/.docs/02_pipelines/training_data_pipeline.md` to see what actually trains.**

**Systematic backstop (Plan 319 pattern):** when multiple training-recipe gaps accumulate from PASS-Redirects across repos, batch them into a `riir-train/.plans/NNN_training_recipe_gap_backlog.md` plan (Plan 319 is the canonical example — it caught GEPO + DGD + CD-LAM + S-TTT + RePlaid LR from katgpt-rs/.research PASS-Redirects in one sweep). Quarterly or when ≥3 gaps accumulate, re-run the audit: grep `PASS-Redirects.*riir-train` across `katgpt-rs/.research/` + `riir-ai/.research/`, check which have no corresponding `riir-train/.plans/` file, batch the stragglers. CMuon (arXiv:2608.02502) was the canonical miss — it came AFTER Plan 319 closed, so the audit didn't catch it; Plan 325 was the retroactive fix.

If Path 0 decomposition fails (the math does NOT decompose into modelless primitives), check whether a training round-trip is affordable under the current Plan 318 GPU regime (~4.7 s/step on 4090 for the 0.40B fixture, ~13 hours for a 10K-step run; the 4B target will be ~10× slower — see Plan 318). If affordable, the deferral is not a "genuine riir-train dependency" but a **cost-justified training dependency** — and the deferral should be downgraded from a one-line redirect to a **Plan** (in `riir-train/.plans/`) that:
1. Names the specific training recipe (paper + section + hyperparams).
2. Estimates the GPU-hours cost (steps × s/step ÷ 3600).
3. Defines a GOAT gate comparing the trained version against the modelless baseline (the trained version MUST measurably beat the modelless baseline on a named metric — quality, latency, or robustness — to justify the training spend).
4. Promotes the trained version ONLY if it beats the modelless baseline; otherwise keeps the modelless path as default and documents the trained artifact as an opt-in feature gate.

This does NOT weaken the modelless-first mandate — it makes the cost-benefit explicit. A modelless primitive that ships at 95% quality is still the right default; a trained primitive that achieves 99% at 13 GPU-hours may be worth the cost for a headline feature (latent CoT via looping is the canonical example — see Plan 318 "GOAT harvest map").

**Dual-track framing (also adopted 2026-08-05):** the research workflow now supports TWO complementary planning tracks:
- **Modelless track (default):** the current §3.5 protocol — Path 0 + Paths 1–3. Produces inference-time primitives in `katgpt-rs` / `riir-ai`.
- **Model-based track (Plan 318 gated):** for papers where the training recipe IS the value, AND Plan 318 makes the training affordable (Path 0.5). Produces training plans in `riir-train/.plans/` that align with the modelless runtime via `LoraPair { reader, writer }` consumption (Plan 025) and freeze/thaw (`MerkleFrozenEnvelope`).

The two tracks are NOT competing — they are complementary. The modelless track produces the runtime; the model-based track produces the trained weights that flow into the runtime via freeze/thaw. A single paper can contribute to BOTH tracks: a modelless inference primitive (shipped now) + a trained weight artifact (shipped when Plan 318 produces a checkpoint). The canonical pattern is LT2 (Research 073 / Plan 108) = modelless looped runtime, + Ouro (Research 073 PASS-Redirect) = trained looped weights that would feed LT2 via freeze/thaw.

**The three modelless unblock paths (check ALL before deferring, AFTER path 0 decomposition fails; run Path 0.5 AFTER all three fail to confirm the training cost is justified):**

1. **Freeze/thaw snapshot correction** (`riir-neuron-db/src/freeze.rs`, `MerkleFrozenEnvelope`) — can a frozen snapshot state, thawed at inference, fix the issue? If the failure is a systematic bias from a runtime construction (e.g., doubled signal, position mismatch, attention pattern asymmetry), a corrected snapshot + thaw may eliminate it without gradient descent.
2. **Raw/lora reader-writer hot-swap** (`LoraPair { reader, writer }`, Plan 025; `LoRAHotSwap`, `dispatch_lora_merge` in riir-ai) — can a **deterministically constructed** (not trained) reader or writer adapter fix the issue? Applying a pre-constructed LoRA overlay is modelless (weight addition, no backprop). The question is: can the correction be derived in closed form (e.g., scale-by-0.5, zero-out-specific-positions, identity-minus-projection) rather than learned via gradient descent?
3. **Latent-space correction** (dot-product projection + sigmoid gate, per constraint #2) — can the bias be corrected by projecting the latent state onto a correction direction and gating the output? This is the modelless analog of a trained adapter: instead of learning the correction, derive it analytically from the failure mode.

**Decision protocol (compact):**

```
Paper appears to need training
  → Path 0: value = MATH (closed-form) not training loop?
    YES → decompose, grep for modelless analog of each component.
      ALL have analogs → MODELLESS-VALIDABLE (fusion). No deferral.
      SOME missing → check paths 1–3 for those.
    NO (value = optimizer/loss/curriculum/RL algorithm) → genuine candidate; still run 1–3.
  → Systematic characterizable cause ("signal doubled", "position offset")?
    YES → path 1 (freeze/thaw)? → path 2 (deterministic LoRA)? → path 3 (latent projection)?
      ALL fail → Path 0.5 (Plan in riir-train if applicable, else redirect with justification).
    NO → Path 0.5 directly.

MODELLESS-VALIDABLE gates MUST be implemented + checked BEFORE any riir-train deferral.
If path 0 + 1–3 all fail → Plan in riir-train (if applicable) or redirect with explicit WHY each failed.
```

**Documentation requirement:** every riir-train Plan MUST include: (1) **Path 0**: what math components were decomposed, which had modelless analogs, which did not; (2) which of paths 1–3 were checked + why each failed; (3) what specifically requires gradient descent that no deterministic construction can provide; (4) **Path 0.5**: whether training is affordable under the current GPU regime — if YES → Plan with recipe + GPU-hours + GOAT gate; if NO → note the cost barrier + lift condition; (5) **Dual-track contribution**: modelless (inference primitive shipped now) vs model-based (training plan) vs both.

### 3.6. Defend-wrong PoC for parity claims — MANDATORY before any "already ships" / "parity" verdict

**Hard rule:** before claiming in a verdict that a paper's mechanism "already ships" modellessly, achieves "parity" with the paper, or that the runtime analog "covers" the paper's loop, you MUST distinguish three claim types and prove each at the level it requires:

| Claim type | Example | Proof required |
|---|---|---|
| **Architectural** ("the runtime analog exists") | "the plan-execute-adapt-replan loop ships as `ReestimationScheduler`" | grep + read the code (sufficient) |
| **Latency / resource** ("modelless, sub-µs, no GD") | "adaptation overhead is +30 ns" | criterion bench |
| **Quality** ("matches / beats the paper's numbers") | "recovers planning success under shift as well as the paper's loop" | **head-to-head PoC on a controlled toy benchmark — architectural reasoning is NOT sufficient** |

**The failure mode this prevents:** claiming all three with only architectural evidence. Architectural coverage does NOT imply quality parity — the shipped version may have tuning gaps, divergence modes, or trigger thresholds that make it underperform the paper on the paper's own task. A grep proves the mechanism exists; it does not prove the mechanism *works as well as the paper's version*.

**When a PoC is mandatory:**
- Any verdict that asserts quality parity ("matches", "competitive with", "recovers as well as", "covers the paper's loop at parity").
- Any Super-GOAT/GOAT claim where the gain is qualitative ("recovers from distribution shift", "matches paper's success rate").
- **Any PASS verdict that downgrades a paper on the grounds that "the runtime analog already ships"** — the downgrade is only honest if the analog actually performs. A PASS verdict backed only by architectural reasoning is the #1 false-PASS failure mode.

**When a PoC is NOT required:**
- Pure architectural redirects (paper X is a refinement of shipped primitive Y, no quality claim).
- Training-only redirects that are genuinely out of scope for our model-based track (→ riir-train one-line redirect, no parity claim). NOTE: training-efficiency papers applicable to our model-based track do NOT get a one-line redirect — they get a Plan in `riir-train/.plans/` per §3.5 Path 0.5, and the Plan's GOAT gate IS the PoC (trained-vs-modelless comparison).
- Latency-only claims (a single criterion bench suffices, no full PoC).
- Low-confident verdicts that explicitly mark the quality claim as unproven and create a `.issues/` entry to track the PoC follow-up.

**Where the PoC lives:** `riir-ai/crates/riir-poc/` — the "defend-wrong" R&D crate. It exists for exactly this: empirical settlements of disputed primitives before any verdict becomes a feature flag. A PoC has three competitors minimum: the paper's mechanism (or its distilled modelless analog), a frozen/no-adaptation baseline, and the shipped runtime analog. Run them head-to-head on a controlled toy domain (no training), print a verdict table. Use `CARGO_TARGET_DIR=/tmp/...` per the AGENTS.md rule and clean up when done.

**The PoC's job is to defend OR refute.** A PoC that only confirms the verdict is weaker than one that honestly refutes part of it. If the PoC refutes the quality claim:
1. **Do NOT silently revise the verdict to match the PoC.** Record the raw numbers in the research note as a §"PoC Addendum" section.
2. **Honest revision:** explicitly state which claim type was confirmed (architectural, latency) and which was refuted (quality). The verdict stands on the confirmed axes; the refuted axis becomes a tracked follow-up (issue in `.issues/`).
3. **The PoC stays as a permanent regression check** in `riir-poc` — its job was to settle the dispute, and it should keep settling it if the shipped primitive is later tuned.

**Canonical example (Research 360, AdaJEPA, 2026-07-01):** verdict claimed "parity" between shipped `ReestimationScheduler` and AdaJEPA's per-MPC-step GD loop based on architectural coverage alone. The PoC at `riir-ai/crates/riir-poc/benches/adajepa_modelless_goat.rs` confirmed latency parity (~940 ns/replan) + architectural coverage, but **refuted quality parity** — the coherence trigger was too conservative for mild shifts, and all adaptation strategies diverged on overshoot shifts. Verdict honestly revised in §9 PoC Addendum; follow-ups tracked in `riir-ai/.issues/363`. **Architectural coverage ≠ quality parity** — grep proved the loop existed, the PoC proved it didn't perform.

### 4. Published prior-art search — MANDATORY before any novelty verdict (the MoTE lesson, Research 411, 2026-08-09)

**This is NOT optional. This is NOT "search if curious". This is a hard gate.** Before claiming ANY novelty (Q1 in §1.5), you MUST search the web for published prior art on the paper's headline technique. The canonical failure: Research 411 (BitNet Ternary MoE) claimed "ternary MoE is novel" — a single web search for "mixture of ternary experts" would have found **MoTE (arXiv:2506.14435, June 2025)** which publishes exactly that technique. The miss happened because the agent only read the paper's own references + grepped the codebase, never searching the broader literature for published work threatening the novelty claim. The user had to manually prompt "search web for more paper on this" to catch it.

**MANDATORY web searches (run ALL before any novelty claim):**
1. Search the paper's **headline technique** verbatim (e.g., `"mixture of ternary experts"`, `"ternary weight MoE"`, `"BitNet distillation recipe"`).
2. Search **2–3 component techniques** (e.g., `"ternary expert up-cycling"`, `"1.58-bit MoE"`, `"sparse ternary routing"`).
3. Search the **selling-point framing** you're about to claim (e.g., `"neuro-symbolic game AI ternary"`, `"cheaper symbolic reasoning LLM"`).
4. Search for **recent (last 2 years) surveys** on the paper's topic — these name the competitive landscape.

**If any published paper does what the note claims is novel → downgrade Q1 (novelty) BEFORE writing the verdict.** Cite the prior art explicitly. Do NOT write the Super-GOAT/GOAT verdict and then discover the prior art in a later pass — that's how mis-citations + overclaimed novelty ship.

**Use subagents to parallelize.** Spawn 2–3 `spawn_agent` calls in parallel: one for the headline-technique web search, one for the component-technique web search, one for the all-repos codebase grep (§1 step 1). The web searches catch published prior art; the codebase grep catches shipped prior art. Both are mandatory; neither substitutes for the other. The Research 411 failure was doing ONLY the codebase grep (which found local notes) and skipping the web search (which would have found MoTE).

**Re-run after corrections.** If you discover prior art in a later pass (user-prompted or self-caught), the refinement is NOT complete until you've: (a) updated the novelty verdict in the note, (b) re-checked whether the surviving novelty claim still holds, (c) committed the correction. Do not leave the note in an overclaimed state.

### 4.5. Optional deeper search

If the mandatory prior-art search (§4) surfaces a rich landscape, use web search for deeper exploration of specific papers, authors, or follow-up work. This is the "curious" tier — not mandatory, but valuable when the prior-art landscape is dense.

## Constraints (non-negotiable)

1. **Modelless first** — inference-time only. No LLM training, no backprop through base weights. Closest to "training" allowed: freeze/thaw snapshot cycles, raw/lora reader-writer hot-swap with **deterministically constructed** adapters (not trained), and latent-space direction-vector updates at runtime. **Before deferring any gate/mechanism to riir-train, exhaust §3.5 modelless unblock paths (freeze/thaw, raw/lora, latent correction).**
2. **Latent-to-latent preferred** — operate in embedding/latent space as long as possible. Decode to tokens or project to raw scalars only at the boundary. Use dot-product + **sigmoid** (never softmax) for projections onto learned direction vectors. Semantic domain (emotion, mood, curiosity, style) → latent. Physical domain (position, HP, wallet balance) → raw, deterministic, synced.
3. **Freeze/thaw over fine-tuning** — the only weight mutation allowed at runtime is swapping a frozen snapshot (atomic, versioned, BLAKE3-checked) or applying a deterministically constructed LoRA overlay (raw/lora hot-swap, no gradient descent). Never mutate weights in-place during inference. If a paper needs gradient updates (after exhausting §3.5 modelless paths), redirect to riir-train.
4. **Self-learn / adaptive CoT welcome** — runtime curiosity, latent prediction, trajectory folding, collapse detection. These update latent state / direction vectors / routing tables, NOT base weights.
5. **7-repo discipline** — katgpt-rs (public engine) → riir-ai (private runtime/game) → riir-chain (private chain) → riir-neuron-db (private neuron-shard leaf) → riir-train (private training) + riir-game-sdk (private game-vocabulary facade, downstream of riir-ai) + riir-armageddon (private product-domain types). Keep the commercial strategy intact. Training know-how never leaks to katgpt-rs; chain IP stays in `riir-chain/`, not `riir-ai/`; neuron-shard IP stays in `riir-neuron-db/`, not `riir-chain/` (chain only re-exports via the `neuron_db` feature); SDK stays a facade over `riir-games-shared` (no engine/chain/db direct deps).
6. **SOLID, DRY** — per `katgpt-rs/.contexts/optimization.md`. Zero-allocation hot paths. Pre-computed lookup tables. Fixed-size arrays for bounded domains.
7. **Tests/examples** — before/after showing the gain (latency, quality, or security). For latent ops: show the projection preserves ranking. For freeze/thaw: show readers never see torn snapshots.
8. **CPU/GPU/ANE auto-route** — threshold-adaptive dispatch. Plasma (µs, CPU/SIMD) → Hot (sub-ms, GPU) → Warm/Cold (ms+, GPU/ANE). Latent ops that fit in L1 cache stay on SIMD; manifold ops that need batched matmul go to GPU.
9. **Plasma → Hot → Warm → Cold → Freeze tiering** — aim for perf on game side (plasma/hot latency budget) AND security on chain side (cold/freeze commitment, BLAKE3-hashed, tamper-evident). Latent state that crosses the sync boundary MUST be raw scalars (valence/arousal/desperation/calm/fear), never the full embedding vector.

## Latent vs raw space rules (critical for game AI)

Reinforce these when designing game systems or chain state:

- **Physical domain** (position, velocity, HP, wallet balance): MUST remain raw exact values. Deterministic replay, quorum sync, anti-cheat require bit-identical reconstruction.
- **Semantic domain** (emotion, mood, curiosity, style, habit): SHOULD operate in latent space via dot-product + sigmoid onto learned direction vectors.
- **Social domain** (encounters, relationships, factions): SHOULD produce KG triples from proximity in latent/embedding space, not from raw coordinate distance.

**Sync boundary:** if data flows through `SyncBlock → ChainConsensus` quorum commit → Cold tier, it MUST be raw and deterministic. If data is consumed locally (emotion projection, shard retrieval, consolidation sleep-cycle), it SHOULD be latent. Bridge functions (raw→latent projection, latent→raw scalar clamp) MUST be zero-allocation, gateable by feature flag, and not introduce sync dependency.

**KG triple emission:** semantic encounters → KG triple from latent similarity. Physical events → TxDelta with raw values, NOT KG triple. Never substitute latent embedding for raw position in anti-cheat validation.

**Spatial cognition (two-brain model):** info brain = real `MapPos` (synced, ground truth). Think brain = per-NPC `SpatialBelief` (zone-level KG triple + stale last_known_pos, fog-of-war gated, NOT synced). Bridge is one-way: real position → belief update only when within `visible_radius`. Confidence decay: `sigmoid(-λ * (current_tick - last_observed_tick))`. Two brains MUST exist independently — divergence is emergent behavior, not a bug.

## Cross-references (read on demand)

**Commercial strategy / moat map:**
- The inline short version of the commercial strategy lives in §"Commercial strategy — inline short version" above (7-repo roles, tier model, What/How rule, benchmark exception, asymmetric cognitive moat, why-hard-to-replicate, FV moat). No external doc lookup needed for routing decisions.
- `riir-ai/.docs/README.md` (+ `03_pillars/README.md`, `04_supergoat_candidates/README.md`) — the **live moat map by capability**. Read these for any Super-GOAT novelty gate or "does this become a pillar?" MOAT-gate question (§1.6). The full internal strategy doc with exhaustive moat analysis lives at `riir-ai/.research/003_Commercial_Open_Source_Strategy_Verdict.md` (commercially sensitive — read only when the inline short version is insufficient).
