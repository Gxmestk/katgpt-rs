---
name: research
description: Research workflow for distilling ML/AI papers into modelless inference primitives, freeze/thaw runtime patterns, and latent-space operations across the katgpt-rs / riir-ai / riir-chain / riir-neuron-db / riir-train / riir-game-sdk / riir-armageddon 7-repo stack. Use when reading arxiv papers, deciding which repo a paper belongs in, creating .research/ notes or .plans/ files, implementing modelless inference primitives, or routing training-vs-inference insights. Enforces the 7-repo commercial strategy (public engine / private runtime / private chain / private neuron-db / private training / private SDK facade / private product-domain), modelless-first constraint, latent-to-latent preference, and freeze/thaw-over-fine-tuning rule.
---

# Research Workflow — Modelless Inference, Freeze/Thaw, Latent-to-Latent

Training-method research lives in `riir-train`. This repo (`katgpt-rs`), `riir-ai` (freeze/thaw runtime + self-learn/adaptive NPCs + game systems), `riir-chain` (neuro-symbolic chain transport, LatCal, chain economics), and `riir-neuron-db` (neuron weight shards, BLAKE3/Merkle commitment, freeze/thaw envelope, consolidation, AnyRAG gateway, vibe KG triples) ship **runtime + latent-space operations**. No LoRA training, no adapter fine-tuning, no optimizer research here. If a paper's value is its training loop → `riir-train/.research`. If its value is a latent-space insight, a routing trick, a freeze/thaw pattern, a chain-commitment bridge, a neuron-shard primitive, or a modelless inference primitive → distill here.

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
- `riir-train/` — private training vault. Adapter training, optimizers, loss functions. Out of scope for this workflow — just note "→ riir-train" and stop.

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

**Hard rule:** before any distillation/verdict/file creation, do **all three**:
1. **`read_file` 4 READMEs + `riir-ai/.docs/README.md`** — defines repo purpose, moat map, raw-vs-latent sync boundary. Skipping = #1 cause of false Super-GOAT claims.
2. **`list_directory` all 4 `.research/` folders** (create `riir-chain/.research/` + `riir-neuron-db/.research/` on first use).
3. **`list_directory` 4 runtime/chain/neuron-db src trees** — module names are codebase vocabulary. Skipping = #2 cause of false Super-GOAT claims.

**Reads:** `katgpt-rs/README.md`, `riir-ai/README.md`, `riir-chain/README.md`, `riir-neuron-db/README.md`, `riir-ai/.docs/README.md` (+ `03_pillars/README.md` + `04_supergoat_candidates/README.md` before any Super-GOAT gate — claiming novelty over a moat that ships is the worst false-positive).

**`list_directory`:**
- 4× `.research/` (katgpt-rs, riir-ai, riir-chain, riir-neuron-db)
- `riir-ai/crates/riir-engine/src/` — runtime vocab (`latent_functor/`, `cgsp_runtime/`, `hla/`, ...). Skipping caused the DiPOD miss: `latent_functor/reestimation.rs` ships "drift-triggered self-healing swap" as "coherence-driven re-estimation scheduler".
- `riir-ai/crates/riir-games/src/` — game systems
- `riir-chain/src/` — `encoding/` (LatCal), `consensus/`, `economics/`, `asset_lifecycle/`, `forensic/`, `catchup/`, ...
- `riir-neuron-db/src/` — `shard.rs`, `index.rs`, `merkle.rs`, `freeze.rs`, `mape_k.rs`, `consolidation.rs`, `gateway.rs`, `vibe.rs`, `spectral_flatness.rs`, `shard_compactor.rs`

Do NOT create any file until all three done. Plans (`.plans/`) are grepped during fusion search (§1), not pre-flight.

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
| `katgpt-rs/crates/katgpt-core/src/sense/` | HLA belief-state kernels, `evolve_hla`, `SenseModule::project`, ternary bit-plane projection | Per-NPC recurrent latent state — the runtime substrate for any "hidden state" / "belief" / "activation" paper |
| `riir-ai/crates/riir-engine/src/latent_functor/` | `zone_gating.rs`, `reestimation.rs`, `arithmetic.rs`, `cross_game.rs`, `k_selector.rs`, `quality_gate.rs` | **Game-theory in latent space** — functors as vector ops, coherence-driven re-estimation, zone-gated activation. Maps any "stage" / "application" / "bypass" / "collapse" paper |
| `riir-ai/crates/riir-engine/src/hla/` | `kernel.rs`, `forward.rs`, `types.rs` — per-NPC 8-dim latent state (valence/arousal/desperation/calm/fear + 3) | The emotional/cognitive latent state — maps any "subspace" / "width" / "channel" paper to per-NPC affect |
| `riir-ai/crates/riir-engine/src/cgsp_runtime/` | Curiosity-guided self-play, latent prediction SSL, MCTS collapse bridge | Runtime curiosity/exploration — maps any "self-learn" / "entropy-driven" / "collapse recovery" paper |
| `riir-neuron-db/src/` | `shard.rs` (NeuronShard Pod, `style_weights[64]`, dendritic branch), `freeze.rs` (`MerkleFrozenEnvelope`), `consolidation.rs` (Raven/δ-Mem), `gateway.rs` (AnyRAG escalation), `vibe.rs` (KG triple arch agent), `merkle.rs` (generic MerkleTree/Proof), `mape_k.rs` (self-healing loop), `spectral_flatness.rs` (lottery-ticket init), `shard_compactor.rs` | **Frozen latent-state storage + integrity + retrieval** — the persistence substrate for any "snapshot" / "integrity envelope" / "memory consolidation" / "external knowledge escalation" / "KG triple emission" paper. Maps any "memory" / "replay buffer" / "experience replay" / "spectral init" / "Merkle commitment" paper. |
| `riir-chain/src/encoding/latcal*.rs` + `latcal_fixed.rs` | Lattice Calculus: 2×2 matrix arithmetic obfuscation, fixed-point bridge, spectral fixed-point, batch determinant validation, DeFi programs | **The sync-boundary bridge** — deterministic, committed, raw-numeric. Maps any "fixed-point" / "deterministic commitment" / "raw↔latent bridge" / "arithmetic obfuscation" paper. LatCal is how latent ops become chain-committed raw values. |
| `katgpt-rs/crates/katgpt-core/src/dec/` | `operators.rs` (d=`exterior_derivative`, δ=`codifferential`, Δ=`hodge_laplacian`), `hodge.rs` (`hodge_decompose` exact/coexact/harmonic, `betti_numbers`, `harmonic_projector`), `flow.rs` (`DecFlowField` exact/coexact/harmonic channels), `terrain_cochains.rs` (Safety/Threat/Occupancy/Destruction typed cochains) — **shipped Plan 251, Research 219** | **The Generalized Stokes' Theorem substrate** — `d∘d=0` enforced by construction (tests verify `curl(grad)=0`, `div(curl)=0`). Maps any "divergence" / "boundary flux" / "line integral" / "curl" / "Hodge decomposition" / "Fokker-Planck" / "mass conservation" / "manifold geometry" / "exterior calculus" / "Stokes theorem" paper. The math ships; thin wrapper primitives (`boundary_flux_mass`, `belief_mass_divergence`, `line_integral`) are Plan 314 (Research 296). **Curse-of-dimensionality caveat: boundary-vs-volume wins only for d ≤ 3 (game maps, HLA regions, KG embeddings) — NOT high-dim shards.** |

**Adapter routing, KV compression, and speculative decode are GOAT-tier framings. Latent-to-latent operations on HLA/functor/neuron-shard/LatCal state are Super-GOAT-tier framings. Attempt the Super-GOAT framing first.** Defaulting to adapter routing when a latent-space reframing is stronger is the primary failure mode this protocol prevents.

## Redirect to riir-train (do NOT distill here)

**MANDATORY pre-check:** before redirecting ANY mechanism to riir-train, exhaust the modelless unblock paths in §3.5 below — **starting with Path 0 (training-target decomposition)**. A mechanism that *looks* training-only may be modelless-validable because its training-target MATH decomposes into already-shipped primitives (dllm interpolant + Latent Field Steering reward-gradient + freeze/thaw replay buffer + induced CWM direction mining). The Flow Sampling lesson (arxiv 2605.03984) is the canonical case: "trains a drift network u_θ via backprop" was initially → riir-train; the conditional drift formula actually decomposes into modelless primitives we already ship. Only redirect if §3.5's decision protocol (Path 0 + Paths 1–3) returns "genuine riir-train dependency".

If a paper is genuinely training-only (after §3.5 Path 0 + Paths 1–3 check) → note "→ riir-train" in one line and stop. Do not create files in this session for it.

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
| `riir-train` (private) | Training research vault | **Only if the paper's value is its training method.** Out of scope for this workflow — just note "→ riir-train" and move on. |
| `riir-game-sdk` (private) | Game-vocabulary facade + dev-tool workspace | **Downstream consumer** — typically not a distillation target. Research lands here only when the paper's insight is about SDK-facing vocabulary, builder patterns, or backend abstractions (rare). The vocabulary substrate itself lives in `riir-games-shared` (in riir-ai workspace). |
| `riir-armageddon` (private) | Arena/game-product domain types | **Downstream consumer** — typically not a distillation target. Research lands here only for product-domain type design (rare). |

Distill into:
- **Modelless** → `katgpt-rs/.research/` + `katgpt-rs/.plans/` + `katgpt-rs/src/` (or `katgpt-rs/crates/katgpt-core/`)
- **Runtime/game** → `riir-ai/.research/` + `riir-ai/.plans/` + `riir-ai/crates/`
- **Chain / LatCal / sync-bridge / commitment / quorum / catchup** → `riir-chain/.research/` (create if missing) + `riir-chain/.plans/` + `riir-chain/src/` (or `riir-chain/crates/`)
- **Neuron shards / freeze envelope / consolidation / AnyRAG / vibe KG / Merkle tree / spectral init / shard compaction** → `riir-neuron-db/.research/` (create if missing) + `riir-neuron-db/.plans/` + `riir-neuron-db/src/`
- **Training-only** → note the redirect, do not create files in this session

## Workflow

### 0. Read & classify the paper

Fetch via `https://r.jina.ai/https://arxiv.org/pdf/{ID}` (per AGENTS.md). Ask: *is the value in the training loop itself (optimizer / loss / schedule / RL algorithm), or in the math the training computes (closed-form drift, conditional score, Riemannian correction, steering formula)?* If optimizer/loss/schedule → riir-train. **If the math → run §3.5 Path 0 (training-target decomposition) FIRST**: decompose the math into components and grep for the modelless analog of EACH (see §1 step 2 standing training-math vocabulary) before redirecting. Only if no component has a modelless analog → riir-train.

**Training-math papers are NOT automatically → riir-train (Flow Sampling lesson, arxiv 2605.03984).** A paper framed as "train a drift network u_θ via backprop with replay buffer" was initially → riir-train. Wrong — the value was the **closed-form conditional drift** `u_{t|0} = α̇X_1 + σ̇x_0 + γ∇r(X_1)`, which decomposes into primitives we ALREADY ship: (a) interpolant `α̇X_1 + σ̇x_0` (dllm NoiseSchedule, D2F, ELF), (b) reward-gradient steering `γ∇r` (**Latent Field Steering Plan 309, DEFAULT-ON** — frozen direction vector IS the ∇r analog), (c) replay buffer (freeze/thaw + MAPE-K). **Do NOT claim the Riemannian extension (parallel transport `T_{X1→Xt} = I - 2P_{X1+Xt}` + Jacobian correction) is a "refinement of Slerp" — it is not.** Slerp IS the geodesic on the sphere (already curvature-optimal by construction); the Riemannian machinery is for **transporting tangent vectors along the geodesic to sample from a distribution on the sphere**, a problem we DO NOT HAVE — our vMF usage (Plan 405) is a **deterministic sigmoid confidence gate** via Eq 17 reduction (`δ = -tanh(κ·s_T) = 1 − 2·sigmoid(2κ·s_T)`), not sampling from a vMF density. Research 382 §3.6 explicitly ruled a PoC NOT REQUIRED; the closest fusion (F1: personality drift auto-correction) was evaluated and REJECTED as not-Super-GOAT (failed Q1-Q4; Issue 039, resolved-and-removed). **PTRM cautionary flag (Research 049):** gradient-guided Langevin gave zero gain in PTRM's use case — cite when the fusion touches gradient-guided sampling. **Lesson: Path 0 decomposition is mandatory, but decomposed components must be grep-checked against actual consumers before claiming GAIN — a modelless analog that ships is necessary but NOT sufficient; there must also be a use case that needs it.**

**Hardware / accelerator / NMP / ASIC papers are NOT automatically PASS.** This is the R418 lesson (StreamDQ, arxiv 2607.08993): a paper framed in HBM / RTL / 12nm CMOS / NMP / D2D PHY / thermal-sim vocabulary was initially PASS-ed as "hardware-only, not software-distillable" — wrong. The paper's *value* was the **technique** (LUT INT→FP, shared FP32 ALU, sideband-tag dispatch, S/Z co-location), which is substrate-independent. Our codebase explicitly **simulates hardware dequant via software SIMD** (Research 110 Ciot — Plasma tier = ternary SIMD, Cold tier = Q4_K dequant-on-read). The revised verdict was Gain; the implementation shipped 2.3× speedup (vs a pessimistic 1.0-1.5× prediction).

**Database / systems / storage / agent-infrastructure papers are NOT automatically PASS.** This is the R300 lesson (Trellis / Experience Graphs, arxiv 2606.29823): a paper proposing a SQL/Cypher/Vector fused query planner (Axiom/Velox) over experience graphs was initially PASS-ed as "we don't ship a database engine, building one is out of scope" — wrong. `riir-neuron-db` IS a database (Pod storage + lock-free `ShardIndex` + Merkle commitment + MAPE-K self-healing + Raven/δ-Mem consolidation + `ItemEmbedIndex` + vibe KG triples + migration/backup). The paper's *value* was the **access pattern** (AS-OF temporal query, vector-seeded graph traversal, materialized training views, governed retraction), which is substrate-independent of the specific SQL/Cypher engine. The revised verdict was Gain. **Conflating "we don't have a fused query planner" with "we don't have a database" is the false-PASS root cause.** When you see "database paper" / "systems paper" / "agent memory infrastructure paper", the FIRST question is: "what *access pattern* does the database engine serve?" — not "we don't have a SQL engine, PASS". Grep `riir-neuron-db/src/` (`shard.rs`, `freeze.rs`, `consolidation.rs`, `gateway.rs`, `vibe.rs`, `merkle.rs`, `mape_k.rs`, `item_index.rs`) + `riir-games/src/trial_log.rs` + `riir-engine/src/episode_buffer.rs` + `riir-engine/src/cgsp_runtime/persistence.rs` for the access-pattern analog before any PASS.

**Hard rule — substrate ≠ value:** Before PASS-ing on ANY paper whose abstract contains hardware vocabulary (HBM, ASIC, RTL, NMP, PE, DQB, tensor-core, TPU, FPGA, accelerator, near-memory, in-memory compute, Processing-In-Memory, PIM), you MUST:
1. Ask: "What is the *technique* here, stripped of the hardware substrate?" (e.g., "LUT-based type conversion", "shared ALU across formats", "co-located metadata buffer", "runtime dispatch tag").
2. Grep the codebase for the software-SIMD analog: `simd_*`, `ternary`, `Plasma`, `Cold tier`, `Q4_K`, `quant/`, `LUT`, `gather`, `shuffle`, `bit_cast`, `from_bits`. **Research 110 (Ciot) is the canonical proof we simulate hardware techniques in software — grep it explicitly.**
3. Only if BOTH (a) the technique has no software SIMD analog AND (b) the codebase has no existing simulation pattern → PASS with explicit documentation of both checks.

The same rule applies to **system / kernel / scheduler / OS papers** — the substrate (Linux kernel, io_uring, DPDK, RDMA) is not the value; the technique (lock-free queue, batching, zero-copy, prefetch hint) often is. Translate the substrate before PASS-ing.

### 1. Distill fundamentally — fuse, don't just direct-map

Don't direct-map the paper to our code. Find the transferable primitive: the geometric, spectral, or information-theoretic insight that works without the paper's training setup. **Then look for fusion opportunities**: cross-pollinate this paper's insight with existing `.research/` notes, `.plans/`, and shipped primitives to synthesize a *novel* combination. The highest-value Super-GOATs in freeze/thaw runtime and self-learn/adaptive CoT almost always come from **fusing** 2–3 papers, not from a single-paper direct mapping.

**Fusion examples that shipped (two patterns — cross-repo + multi-primitive):**
- **Cross-repo:** Gemini Fourier × LatCal → `katgpt-rs/.research/212_*` → `katgpt-rs/.plans/242_*` (a `katgpt-rs` modelless primitive fused with a `riir-chain` commitment bridge)
- **Multi-primitive:** Collapse-aware × bandit coverage × sigmoid margin → `katgpt-rs/.plans/212_*` × `157_*` × `061_*` (three inference primitives fused into one collapse-recovery gate)

**Fusion protocol:**
1. **MANDATORY — grep ALL SEVEN repos in this session, BOTH layers (notes AND code). Do NOT stop after the first repo or the first layer.** Run keyword / paper-title / author / primitive-name grep across:
   - `katgpt-rs/.research/` + `katgpt-rs/.plans/` (intent — what we planned)
   - `riir-ai/.research/` + `riir-ai/.plans/` (intent — runtime/game)
   - `riir-chain/.research/` + `riir-chain/.plans/` (intent — current chain research; `.research/` may need creating on first use)
   - `riir-neuron-db/.research/` + `riir-neuron-db/.plans/` (intent — current shard research; `.research/` may need creating on first use)
   - `riir-ai/.docs/` — the consolidated selling-point book (`pillars/`, `04_supergoat_candidates/`, `reasoning/`, `self_learn_npcs/`, `neuro_symbolic_chain/`, ...). These are not academic distillation; they are the moat/selling-point framing. Grep them alongside `.research/` so you do not claim novelty over a pillar that already ships.
   - `katgpt-rs/src/` + `katgpt-rs/crates/` (shipped primitives — what actually exists)
   - `riir-ai/crates/` (shipped runtime)
   - `riir-chain/src/` + `riir-chain/crates/` (shipped chain — LatCal, encoding, economics, forensic, catchup, etc.)
   - `riir-neuron-db/src/` (shipped shards — `shard.rs`, `freeze.rs`, `consolidation.rs`, `gateway.rs`, `vibe.rs`, `merkle.rs`, `mape_k.rs`, `spectral_flatness.rs`, `shard_compactor.rs`)
   - **Super-GOAT factory modules** (from §Primary focus) — `list_directory` these explicitly even if the paper looks pure-training: `katgpt-rs/crates/katgpt-core/src/sense/`, `riir-ai/crates/riir-engine/src/latent_functor/`, `riir-ai/crates/riir-engine/src/hla/`, `riir-ai/crates/riir-engine/src/cgsp_runtime/`, `riir-neuron-db/src/` (shards/freeze/consolidation/AnyRAG/vibe/merkle), `riir-chain/src/encoding/latcal*.rs`, `katgpt-rs/crates/katgpt-core/src/dec/` (Stokes/exterior-derivative/Hodge — maps any divergence/boundary/line-integral/Fokker-Planck/manifold-geometry paper)

   (riir-train is deliberately excluded — training methods are out of scope for this workflow.)

   Two layers, seven repos (5 primary distillation targets + 2 downstream consumers). The closest cousin is frequently in the OTHER repo (e.g., a `katgpt-rs` modelless primitive fused with a `riir-chain` LatCal commitment bridge — see Gemini Fourier × LatCal; or a `riir-neuron-db` freeze envelope fused with a `riir-ai` runtime adapter hot-swap) OR in the CODE not the notes. **Notes describe intent; code describes what shipped.** A mechanism can ship without a research note — e.g., HLA's `evolve_hla` (`katgpt-rs/crates/katgpt-core/src/sense/reconstruction.rs`) is a per-NPC recurrent belief-state kernel with no `.research/` note framing it as such; a notes-only grep misses it and produces a false Super-GOAT claim (verdict then has to be revised down). If you only grep `katgpt-rs/.research/`, you will miss both axes and produce a duplicate, weaker note, or an overclaimed verdict.

2. **MANDATORY — vocabulary translation before grepping.** Papers and our codebase use different words for the same mechanism. Before any grep, list the paper's 3–5 key mechanism terms, then for EACH, brainstorm ≥2 codebase-equivalent terms by asking: "if we shipped this, what would we call it?" Then grep BOTH sets.

   **Standing latent-state vocabulary (ALWAYS include, even for non-latent papers — most architecture/training papers have a latent-space reframing):**
   - "residual stream" / "hidden state" / "activation" → "HLA state", "belief state", "latent subspace", "sense projection"
   - "layer" / "depth" / "stage" → "decision stage", "functor application", "cgsp cycle", "consolidation tick"
   - "width" / "dimension" / "capacity" → "latent subspace", "active projection channel", "sense channel"
   - "carry-forward" / "bypass" / "skip" → "leaky integrator", "dormant subspace", "decay gate", "persistence"
   - "collapse" / "degeneration" / "valley" → "coherence decay", "re-estimation trigger", "staleness"
   - "bottleneck" / "narrowing" → "subspace projection", "channel selection", "zone gating"
   - "fixed-point" / "deterministic" / "committed" → "LatCal", "lattice calculus", "BLAKE3 commitment", "raw scalar bridge"
   - "divergence" / "flux" / "∇·F" / "density change" → "codifferential", "δ", "DEC divergence", "belief_mass_divergence"
   - "curl" / "vorticity" / "∇×F" / "circulation" → "d₁", "DEC curl", "exterior_derivative rank 1→2"
   - "boundary" / "∂M" / "frontier" / "perimeter" → "exterior_derivative", "d", "coboundary operator", "boundary_flux_mass"
   - "line integral" / "trajectory energy" / "path cost" / "geodesic cost" → "line_integral", "edge field sum", "rank-1 cochain path sum"
   - "Stokes theorem" / "divergence theorem" / "Green's theorem" / "Gauss" / "generalized Stokes" → "DEC identity d∘d=0", "curl(grad)=0", "div(curl)=0", "hodge_decompose"
   - "Hodge decomposition" / "exact/coexact/harmonic" / "Helmholtz" / "conservative/solenoidal" → "hodge_decompose", "DecFlowField", "exact_flow/coexact_flow/harmonic_flow"
   - "Fokker-Planck" / "continuity equation" / "mass conservation" / "probability flow" → "belief_mass_divergence", "codifferential on belief cochain"
   - "cell complex" / "mesh" / "simplicial" / "cubical" → "CellComplex", "CochainField", "grid_2d"

   **Standing per-NPC runtime / freeze-thaw / personality vocabulary (ALWAYS include when the paper touches per-entity state, memory, personality, swap, evaluator/judge/critic, or selective erasure/forgetting — the `riir-ai/.research/` guide corpus is SATURATED in this space, so paper-vocabulary-only greps produce false novelty claims; see R320 failure below):**
   - "personality swap" / "personality drift" / "character shift" / "persona change" → "committed personality", "freeze/thaw cadence", "direction vectors versioned via freeze/thaw", `ArchetypeBlendShard`, `KarcShard`
   - "selective forgetting" / "memory erasure on swap" / "dependent record invalidation" / "forget on swap" → "non-interference branches", "branch-local", `BranchBank`, "orthogonal subspace projection"
   - "survives swap" / "invariant to observation gaps" / "replay-deterministic personality" → "sampling invariance", "quorum-verifiable", "bit-identical across nodes", "FAME Proposition 3"
   - "epoch boundary" / "controlled utility evolution" / "checkpoint replacement" / "non-stationary utility" → "freeze/thaw cadence", "consolidation sleep-cycle", `tau_reest`, "re-estimation trigger", "coherence < tau"
   - "co-evolution" / "evaluator replacement" / "moving target" → "personality divergence", "direction vector drift", "emergent personality at crowd scale"
   - "evaluator" / "judge" / "critic" / "verifier" (per-entity) → "claim verifier", "CLR vote", "direction vector projection", "Salience Tri-Gate", `ConstraintPruner`
   - "frozen snapshot" / "frozen artifact" (generic) → NAME THE CONCRETE TYPE: `KarcShard`, `ArchetypeBlendShard`, `BranchBank` snapshot, `ZoneGeometryPod`, `MerkleFrozenEnvelope`, `SleepAnticipationShard`. Never use the generic "frozen snapshot" in a selling-point sentence — grep `riir-neuron-db/src/shard.rs` + `riir-neuron-db/src/freeze.rs` for the concrete subtype.
   - "cache invalidation on swap" / "dependent records" / "criterion consistency" → `DecCache::mark_face_destroyed`, `ZoneGeometryCache::invalidate(zone, new_version)`, `topology_version` bump, `SourceShardHashMismatch`

   **Standing compute-unit translation (MANDATORY for agent/LLM papers — the R368 lesson):** Papers increasingly use "LLM forward pass" / "LLM call" as the compute unit for a decision. Our codebase uses different compute units for the same decisions. ALWAYS translate the compute unit, not just the semantic name:
   - "LLM decides what to write/record" → `SpeculativeGenerator` (draft) + `ScreeningPruner` (relevance) + `ConstraintPruner` (validity)
   - "LLM decides what to retrieve/read" → `SpeculativeGenerator` + `ScreeningPruner` + AnyRAG escalation gate
   - "LLM judges/verifies/critiques a claim" → CLR vote + SalienceTriGate + Claim Rubric L1/L2/L3
   - "LLM reviews trajectory + rewrites code/prompts" → Raven/δ-Mem consolidation + MAPE-K self-healing (architectural analog; quality parity needs PoC per §3.6)
   - "meta-LLM generates novel semantic content" → **NO modelless analog** — genuine NO-GAIN if the value IS the generation

   **Standing substrate-translation vocabulary (MANDATORY for hardware / accelerator / NMP / PIM / ASIC / system papers — the R418 lesson):** Papers framed in hardware vocabulary (HBM, RTL, CMOS, NMP, PE, DQB, TPU, FPGA, accelerator) describe techniques whose *implementation substrate* is hardware, but whose *value* is substrate-independent. Our codebase explicitly **simulates hardware dequant / accelerator techniques in software SIMD** (Research 110 Ciot — Plasma tier = ternary SIMD, Cold tier = Q4_K dequant-on-read). ALWAYS translate the substrate, not just the semantic name. Before PASS-ing on a hardware paper, grep BOTH vocabulary sets:
   - "DeQuantization Block (DQB)" / "near-memory processing unit" / "processing element" → "SIMD dequantize function", `dequant_via_lut`, "LUT lookup", `simd_lut_dequant`
   - "HBM base die" / "pseudo-channel" / "memory controller" → "L1 cache", "register file", "SIMD lane"
   - "pseudo-channel-aware layout" / "channel-interleaved" → "cache-line-aligned layout", `AlignedWeightMatrix`, `channel_simd`, "struct-of-arrays", "64-byte rows"
   - "wire-mapping FP-to-FP" / "2:1 mux type conversion" → "bit-cast", `f16::from_bits`, `from_bits`, "SIMD shuffle/permute", `vrev32`, `_mm_shuffle_epi32`
   - "LUT-based INT-to-FP" / "sign-bit extension" / "zero-padded LUT" → "pre-computed lookup table", `[f32; 16]`, `[f32; 256]`, "SIMD gather", `_mm_i32gather_ps`
   - "sideband tag" / "3-bit tag" / "dequantization mode select" → `QuantFormat` enum, "runtime dispatch", "function pointer", `match` on format
   - "S/Z buffer" / "scaling-factor cache" / "group-index calculator" → "L1-resident scale cache", `BlockQ4K { d, dmin, scales, qs }`, "per-group scale"
   - "fused DQ-GEMM kernel" → "fused dequantize-matmul", `simd_dot_f16_f32`, "dequant-in-register"
   - "split DQ-GEMM kernel" → "two-step dequant + dot", `dequantize_row_q4_k` + `simd_dot_f32`
   - "kernel selection" / "batch threshold" → "breakeven routing", `breakeven/`, Plan 218, "memory-bound vs compute-bound"
   - "tensor core" / "CUDA core" / "SM" → "CPU SIMD lane", "FPU", "NEON/AVX2 lane" (the *role*, not the hardware)
   - "io_uring" / "DPDK" / "RDMA" / "zero-copy kernel bypass" → "zero-allocation hot path", `_buf` pattern, `pool.rs`
   - "TPU systolic array" / "PE mesh" → "blocked matmul", `simd_matmul_rows`, "tiled GEMM"

   **Unified decision rule — substrate-as-instantiation vs mechanism-as-value (prevents false-PASS/false-redirect — covers R418 hardware + R368 LLM + R300 database + Flow Sampling training-math):**
   - **Value = technique/access-pattern/decision-structure/math** (not the substrate) → substrate is one *instantiation* → translate to our substrate → GOAT/Gain candidate.
     - **R418 hardware:** LUT lookup / shared ALU / co-located layout / dispatch tag (StreamDQ — DQB-in-HBM is instantiation, SIMD dequant is ours; shipped 2.3×).
     - **R368 LLM:** decision structure (AutoMem LOG/PLAN — LLM is instantiation, probe/draft/pruner is ours). **When you see "N LLM calls/step", FIRST ask "what decision is each call computing?" — not "violates 20Hz budget, NO-GAIN".** R169 guard applies ONLY to genuine LLM-dependent process (semantic code generation, NL verification) — NOT to decision structures.
     - **R300 database:** access pattern (Trellis — Axiom/Velox is instantiation, `riir-neuron-db` Pod + Merkle + vibe KG is ours). `riir-neuron-db` IS a database; prefer latent-first + neuro-symbolic translations over `AnyRAG` external escalation.
     - **Flow Sampling training-math:** closed-form drift / conditional score / Riemannian correction / regression target (training loop is instantiation, dllm + Latent Field Steering + freeze/thaw is ours).
   - **Value = substrate-fabrication-advance itself** (new transistor geometry, new query optimizer algorithm, new optimizer like Muon, new loss function, new RL algorithm, semantic code generation) → no modelless analog → PASS or → riir-train.

   **Standing database-substrate translation vocabulary (MANDATORY for database / systems / storage / agent-state / query-planner papers — the R300 lesson):** Papers framed in database-engine vocabulary (SQL, Cypher, Axiom, Velox, query planner, materialized view, OLTP/OLAP, vector store, experience graph, bi-temporal) describe **access patterns** whose *implementation substrate* is a database engine, but whose *value* is the access pattern itself. Our codebase ships a **database substrate** (`riir-neuron-db`: Pod storage + lock-free `ShardIndex` + Merkle commitment + compaction + MAPE-K self-healing + Raven/δ-Mem consolidation + `ItemEmbedIndex` + vibe KG triples) WITHOUT a SQL/Cypher engine. **Conflating "we don't have a fused query planner" with "we don't have a database" is the R300 false-PASS root cause.** ALWAYS translate the access pattern to **latent-first + neuro-symbolic** primitives. Before PASS-ing on a database/systems/agent-memory-infrastructure paper, grep BOTH vocabulary sets:
   - "experience graph" / "reward-bearing search tree" / "node with parent + fitness + UCB" → `NeuronShard` + lineage fields, `TrialRecord` (`riir-games/trial_log.rs`), `KarcShard`, `ArchetypeBlendShard`, `EpisodeBuffer`
   - "materialized view" / "training-data extraction" → `consolidation.rs` Raven/δ-Mem sleep-cycle output, `ShardCompactor` cold-tier product
   - "vector-seeded graph traversal" / "ANN + relational + graph" → `ItemEmbedIndex::query` (latent cosine ANN) **composed with** `vibe.rs` KgTripleTemplate edges (neuro-symbolic graph). **Latent-first**: the cosine seed IS the latent op; the KG edges ARE the neuro-symbolic structure. `AnyRagGateway` is **optional + preprocess-only** external escalation — NOT the primary retrieval path; do not lean on it.
   - "AS-OF temporal query" / "bi-temporal" / "time travel" / "reconstruct state at step T" → `MerkleFrozenEnvelope` version chain, `PersonalityLedger` ring buffer, `AuditLog` backward replay
   - "frontier selection" / "UCB query" / "visit_count + reward" → `BanditPruner`, `ManifoldBandit` (Plan 370), `MCTS`, `GZeroPlayer`
   - "governed view maintenance" / "retraction propagation" / "invalidate derived" → `MapeKLoop` self-heal cascade, `BackupMirror::integrity_check_3way`, `ChainDivergenceDetector`
   - "change log" / "CDC" / "logical step number" → `PersonalityLedger` ring, `AuditLog` entries, `TrialLog` hash chain
   - "concurrent tree search atomicity" / "MCTS backprop race" → `FunctorEntry` atomic Arc-swap (`latent_functor/table.rs`), `LoRAWeightVersion` lock-free read (`episode_buffer.rs`)
   - "episode buffer" / "experience replay" → `EpisodeBuffer` FIFO (`episode_buffer.rs`), Raven/δ-Mem consolidation input queue
   - "unified query planner" / "SQL/Cypher/Vector fused" → **NO direct analog** (we don't ship a database engine) — BUT individual access patterns map to **latent ops** (cosine, dot-projection) + **neuro-symbolic edges** (KG triples) + **raw commitment** (BLAKE3/Merkle). The planner is the paper's instantiation; the access patterns are ours.

   **Decision rule — database-engine-as-substrate ≠ database-access-pattern-as-mechanism:** See the **unified decision rule above**. `riir-neuron-db` IS a database; the gap is individual access patterns, NOT the substrate. Prefer latent-first (cosine + dot-projection) + neuro-symbolic (KG triples) translations over `AnyRAG` external escalation (optional + preprocess-only).

   Example (DiPOD paper → riir-ai code): paper-vocabulary grep misses `latent_functor/reestimation.rs` which ships DiPOD's "interleave self-distillation when ELBO drifts" as "coherence-driven re-estimation scheduler when coherence < tau_reest". **Vocabulary translation is the only defense — notes framing can use codebase vocabulary that paper-vocabulary grep misses on BOTH layers.**

   Example (Stokes paper → katgpt-rs DEC code): paper-vocabulary grep for `stokes|divergence theorem|fokker-planck` returns ZERO hits. Codebase-vocabulary grep for `codifferential|exterior_derivative|hodge_decompose|DecFlowField` hits `dec/operators.rs`, `dec/hodge.rs`, `dec/flow.rs`. The Generalized Stokes' theorem machinery ships as DEC operators but no note framed it in Stokes vocabulary.

3. **MANDATORY — latent-space reframing before verdict.** Before any verdict, re-cast the paper's core mechanism as a latent-to-latent operation on the codebase's latent-state kernels (the seven Super-GOAT factory modules above). Ask explicitly: "How does this mechanism look when operating on (a) HLA's per-NPC latent state, (b) `latent_functor/` operations, (c) `cgsp_runtime/` curiosity signals, (d) LatCal fixed-point commitment (in `riir-chain/src/encoding/`), (e) `NeuronShard` style_weights / dendritic branch / `MerkleFrozenEnvelope` / Raven consolidation / AnyRAG escalation (in `riir-neuron-db/src/`), (f) DEC Stokes-calculus operators (`katgpt-rs/crates/katgpt-core/src/dec/` — `exterior_derivative` d, `codifferential` δ, `hodge_decompose`, `DecFlowField` exact/coexact/harmonic)?" If your fusion idea only touches adapter routing / KV compression / speculative decode without a latent-state reframing, you are likely in GOAT territory and have probably missed the Super-GOAT angle. If you find yourself reaching for an adapter-routing framing, treat it as a symptom that the stronger latent-functor / HLA / neuron-shard / LatCal reframing is still unfound — adapter routing is the fallback, never the primary Super-GOAT framing. The latent reframing is mandatory even for papers that look pure-training/architecture — most have a latent subspace / stage-gating / persistence / memory-consolidation / manifold-geometry angle that lands in HLA/functor/neuron-shard/DEC.

4. **Zero grep hits ≠ novelty.** If your paper-vocabulary grep AND your codebase-vocabulary grep BOTH return zero hits, that is evidence of one of three things, in order of likelihood: (a) you are still using the wrong vocabulary — try a third semantic angle (e.g., grep for the *output behavior* like "swap when X" instead of the *mechanism name* like "tightness monitor"); (b) the mechanism is genuinely not shipped; (c) the mechanism is novel. Do NOT jump to (c). Default to (a): re-grep with at least one more semantic alternative before claiming "no prior art".
5. After finding the transferable primitive of *this* paper, list the 2–3 closest existing notes/plans **across all seven repos** and ask: "what novel combination of this paper + note A + note B produces a capability none of them has alone?" Write that combination into the research note's §Distillation as a **Fusion** subsection, even if you don't plan it yet.
6. Verdict by the commercial strategy tiers (see §Cross-references for the strategy doc): **Super-GOAT** > GOAT > Gain > Pass (see §Verdict tiers below). **A fusion that produces a new capability class is a strong Super-GOAT candidate — check the novelty gate (§1.5).**
7. Create research `.md` at the right repo (see table above).

**File naming:** `{NNN}_{Short_Title_with_Underscores}.md` where NNN is the next free number (zero-padded to 3 digits, e.g. `239_`, `240_`). Check the folder first — numbers may be non-contiguous; pick the next free slot.

**Research note format** (see `katgpt-rs/.research/238_LoRA_Muon_Spectral_Low_Rank_Manifold.md` for a canonical example):

```markdown
# Research NNN: <Title>

> **Source:** <paper title + arxiv link + authors + date>
> **Date:** YYYY-MM-DD
> **Status:** Active | Done | Shelved
> **Related Research:** NNN (short note), ...
> **Related Plans:** NNN (short note), ...
> **Cross-ref (riir-ai / riir-chain / riir-neuron-db):** Research NNN, Plan NNN   ← only if cross-repo (game runtime → riir-ai; chain/LatCal → riir-chain; shards/freeze/consolidation/AnyRAG/vibe → riir-neuron-db)
> **Classification:** Public | Private   ← katgpt-rs notes are always Public

---

## TL;DR

<2-4 sentences: the distilled primitive, why it matters here, what it unblocks>

**Distilled for katgpt-rs (modelless, inference-time):**
<the transferable insight, stripped of training setup>

---

## 1. Paper Core Findings
...
## 2. Distillation
...
## 3. Verdict

**Tiers (high → low):**

| Tier | Criteria | Routing |
|------|----------|--------|
| **Super-GOAT** | Novel mechanism (no prior art) + new capability class + product selling point + force multiplier (≥2 pillars). Creates a moat. | Open primitive → katgpt-rs. **Architectural guide → riir-ai/.research/ (game runtime) OR riir-chain/.research/ (chain/LatCal) OR riir-neuron-db/.research/ (shards/freeze/consolidation/AnyRAG/vibe/Merkle)**. Plans → appropriate repo(s) as needed. |
| **GOAT** | Provable gain (latency/quality/security) over existing approach, but not a new class of capability. Promotes to default if it wins. | Plan + implement → appropriate repo. Feature flag + benchmark. |
| **Gain** | Incremental improvement, useful but not headline-worthy. | Plan only, behind feature flag. |
| **Pass** | Mechanism already ships OR not relevant OR training-only (→ riir-train). | **No files.** Verdict + one-line reason + closest shipped cousin, in conversation only. If the paper has actionable improvements (config contradiction, unmitigated failure mode, unblocker), that's **Gain**, not Pass — see §1.55. |

**One-line reasoning required for each verdict.** For Super-GOAT: state the selling point explicitly.

**After the tier verdict, run the MOAT gate per domain (§1.6)** — a tier verdict without a domain-fit check can land a great primitive in the wrong repo and dilute the moat. The tier answers "how strong?"; the MOAT gate answers "does this strengthen THIS repo's moat, and is this the right repo?".
```

### 1.5. Novelty gate — is this Super-GOAT?

Before planning, score novelty. Ask all four:

1. **No prior art?** Grep `.research/` + `.plans/` across all repos AND grep the shipped code (`katgpt-rs/src/`, `katgpt-rs/crates/`, `riir-ai/crates/`, `riir-chain/src/`, `riir-chain/crates/`, `riir-neuron-db/src/`) for the primitive name and mechanism keywords. **You MUST grep BOTH paper vocabulary AND codebase-vocabulary alternatives (see §Workflow fusion protocol step 2 — vocabulary translation).** **Notes describe intent; code describes what shipped.** A mechanism can ship under either of two failure modes:
   - **No notes framing at all** — canonical example: HLA's `evolve_hla` (`katgpt-rs/crates/katgpt-core/src/sense/reconstruction.rs`) is a per-NPC recurrent belief-state kernel with no `.research/` note framing it as such; missing it has historically caused false Super-GOAT claims.
   - **Notes framing uses different vocabulary than the paper** — canonical example: DiPOD's "interleave self-distillation when ELBO drifts" is shipped as `riir-ai/crates/riir-engine/src/latent_functor/reestimation.rs` "coherence-driven re-estimation scheduler when coherence < tau_reest". The note DOES frame the mechanism, but using codebase vocabulary, so a paper-vocabulary grep misses it on BOTH notes AND code layers. This is strictly worse than the `evolve_hla` failure: even a diligent notes grep fails. **Vocabulary translation (fusion protocol step 2) is the only defense.**
   If the code already covers the mechanism → not novel, Gain at best. **This three-layer check (notes + code + vocabulary translation) is mandatory — notes-only is the #1 cause of false Super-GOAT claims; paper-vocabulary-only is the #2 cause; skipping the seven Super-GOAT factory modules is the #3 cause.**

   **Grep returns candidates; READING the candidates is mandatory.** A grep hit is a lead to follow, not a prior-art confirmation. When a grep hit's filename or first-line summary touches the candidate's selling-point space (per-NPC, memory, personality, swap, freeze/thaw, evaluator, critic, curiosity, test-time scaling, sleep-time, sub-goal), `read_file` the hit's TL;DR + §1 (selling point) BEFORE claiming novelty. Grepping `riir-ai/.research/`, seeing a filename match, and moving on is the failure mode: the guide frames the mechanism under different vocabulary, so the filename looks unrelated even though the content is exact prior art. **When the candidate selling point touches per-NPC + memory + personality + swap, the `riir-ai/.research/` corpus is saturated — grep it for `Per_NPC|Committed|Cognitive_Branch|Sub_Goal|Karc|CLR|Sleep_Time|Gain_Cost|Personality|Curiosity|Mind_Reading` and READ every hit's TL;DR before claiming novelty. Assume covered until proven otherwise.**
2. **New class of behavior?** Not better numbers, but something no incumbent can do (a new capability, not an optimization).
3. **Product selling point?** Can you finish the sentence: "Our NPCs/systems do X that no competitor can"? If you can't → Gain.
4. **Force multiplier?** Connects to ≥2 existing pillars/systems (check connection map in `.research/`). Solo novelty without integration = GOAT, not Super-GOAT.

**If YES to all 4 → verdict = Super-GOAT.** Mandatory outputs:
1. **Open primitive** → `katgpt-rs` (generic math, no game semantics).
2. **Architectural GUIDE** → the private selling-point doc. **Pick the repo by where the selling point lives**: `riir-ai/.research/NNN_*.md` for game-runtime / HLA / functor / self-learn selling points; `riir-chain/.research/NNN_*.md` for chain / LatCal / commitment / quorum / catchup / sync-bridge selling points (create folder on first use); `riir-neuron-db/.research/NNN_*.md` for shard / freeze envelope / consolidation / AnyRAG / vibe KG / Merkle tree / spectral init / shard compaction selling points (create folder on first use). If the selling point spans multiple repos (e.g., latent ops that cross the chain sync boundary via a shard commitment), create the primary guide in the repo that owns the boundary being crossed, and cross-reference from the others. The guide MUST include:
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

**The rule is simple: PASS = no files. Gain = files.**

Before any verdict, scan the paper for actionable improvements to our stack (config changes, PoC tasks, unblockers, unmitigated failure modes). Then:

- **If the mechanism ships AND there are actionable improvements** → verdict is **Gain**, not Pass. Create `.issues/` entries (per AGENTS.md: "Create issue at .issues for poc, proof, optimization or refactor task, do not create plan") and/or a plan behind a feature flag.
- **If the mechanism ships AND there are no actionable improvements** → verdict is **Pass**. No files. Report verdict + one-line reason + closest shipped cousin in conversation only.
- **If the mechanism does not ship AND it's modelless** → verdict is **Gain** or higher.
- **If the mechanism does not ship AND it's training-only** → note "→ riir-train" and stop.

There is no "PASS-with-gains". There is no middle tier. If the paper produces something actionable, it's a Gain. If it doesn't, it's a Pass.

**What counts as actionable (Gain):**
- The paper's data **contradicts** a current config default, and grep confirms the current value.
- The paper exposes a failure mode in shipped code with **no existing mitigation**.
- The paper unblocks a known deferred task or open issue.

**What does NOT count as actionable (Pass):**
- "Validates our design" / "design principle" / "theoretical lens" — the design stands without it.
- "Empirical validation target" where the paper confirms we were right — not urgent.
- "Composition schedule" / "regime insight" / "compute savings evidence" — interesting but doesn't change code.
- "Could inform a future config" (speculative, no current contradiction).

If unsure whether something is actionable → it's not. Default to Pass.

### 1.6. MOAT gate per domain

The global verdict tiers (Super-GOAT / GOAT / Gain / Pass) measure *how strong* a contribution is. The **MOAT gate per domain** measures *whether a contribution strengthens THIS repo's moat*. A primitive can be a clean GOAT win yet contribute nothing to the moat if it lands outside the repo's pillar scope, or if a stronger latent reframing was missed. **Check the domain MOAT gate at verdict time — a mismatch means reroute to the correct repo.**

| Domain | MOAT contribution bar | In scope | Out of scope (reroute) |
|--------|----------------------|----------|----------------------|
| **`katgpt-rs`** (public engine) | **Paper-derived fundamental / principle / base-foundation primitive** that passes GOAT or Gain via fusion, with **promote/demote tracked per stack**. Aim: research-grade primitives the adoption funnel depends on. Each primitive ships behind a feature flag; the GOAT gate decides promote-to-default vs demote-loser per stack. | Transformer stack (layers, attention, KV cache, sampling, sparse / quant-aware **inference**, speculative decode, DDTree, MCTS, bandits, ConstraintPruners); **2D toy benchmark games** (bomber/go/monopoly/fft-arena) + their generic MCTS/bandit/CCE wiring; DEC/Stokes substrate; HLA kernel; sigmoid mechanics. | Product game wiring (→ riir-ai); chain commitment (→ riir-chain); shard internals (→ riir-neuron-db); trained weights (→ riir-train). |
| **`riir-ai`** (private runtime) | **Pillar-level or Super-GOAT**: fusion-GOAT / fusion-Gain that connects to ≥2 pillars, OR a new pillar candidate (sloppy-test winner). | **Adaptive / self-learn NPCs**, **reasoning pack** (P8), **MMORPG-scale** (20Hz tick, fog-of-war, zone attention, crowd MCGS), **3D game wiring**, freeze/thaw runtime, latent-to-latent ops on HLA/functor/cgsp state. | Generic transformer mechanics (→ katgpt-rs); chain transport (→ riir-chain); shard storage (→ riir-neuron-db); training methods (→ riir-train). |
| **`riir-chain`** (private chain) | **Pillar-level or Super-GOAT**: pillar 3 (riir-chain) amplifier, OR sync-boundary bridge novelty. | LatCal commitment, quorum/catchup, chain economics, asset lifecycle / forensic, DeFi programs, `riir-chaind`, the raw↔latent sync-boundary bridge. | Generic fixed-point math without commitment semantics (→ katgpt-rs); shard internals (→ riir-neuron-db). |
| **`riir-neuron-db`** (private shards) | **Pillar-level or Super-GOAT**: pillar 2 (riir-neuron-db) amplifier, OR shard/freeze/consolidation novelty. | `NeuronShard` layout, freeze/thaw envelope, Raven/δ-Mem consolidation, AnyRAG escalation, vibe KG triples, Merkle integrity, spectral init, shard compaction, dendritic branch. | Chain commitment of shards (→ riir-chain); runtime adapter swap (→ riir-ai). |
| **`riir-train`** (private training) | **Secondary moat**: training-method implementations + configs + trained weight assets (GPU-hours). Out of scope for THIS workflow — note "→ riir-train" and stop. | Adapter training, optimizers, loss functions, quant-aware **training**, DPO/GRPO/SFT pipelines, trained weight assets. | Inference-time / runtime / latent ops (→ katgpt-rs or riir-ai). |

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
- The plan is training-only (→ riir-train).

Invoke via the `skill` tool with name `goat-audit`. The skill's three-layer grep (feature-name + struct/function-name + consumer-vs-duplicate) catches both false negatives (Issue 003's `salience_tri_gate` miss) and false positives (Issue 019's `KVCache` local-shadow duplicate flagged as wired).

### 2. If gain (or GOAT), plan it

Add plan `.md` to `katgpt-rs/.plans/` (modelless), `riir-ai/.plans/` (runtime/game), and/or `riir-chain/.plans/` (chain / LatCal / neuron_db). Use `## Phase N` sections with `- [ ]` per task (mark `- [x]` when done). **Never** plan into `riir-train` from this workflow.

> Super-GOAT plans should be created AFTER the riir-ai guide. The guide is the strategy; the plan is the execution.

**Plan format** (see `katgpt-rs/.plans/271_attention_matching_compaction.md` for a canonical example):

```markdown
# Plan NNN: <Title>

**Date:** YYYY-MM-DD
**Research:** [katgpt-rs/.research/NNN_*.md](../.research/NNN_*.md)
**Source paper:** [arxiv ID.NNN](https://arxiv.org/abs/ID) — <short cite>
**Target:** `katgpt-rs/src/<module>/` (new module) + Cargo feature `<feature_name>`
**Status:** Active — Phase N <state>

---

## Goal

<one paragraph: what ships, what it enables, GOAT gate>

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [ ] **T1.1** <concrete, verifiable task>
- [ ] **T1.2** ...
```

**GOAT gate rule** (AGENTS.md): every plan that introduces a new technique must have a feature flag and a benchmark proving the gain before promoting to default. Demote the loser if the new technique wins.

**UQ-bearing primitive GOAT gate extension (the "Report the Floor" rule, adopted 2026-06-28 per Research 322):** Any primitive that claims a probability distribution, predictive interval, quantile, coverage guarantee, confidence score, or calibrated uncertainty (collectively: **UQ-bearing**) MUST include a benchmark against the **conformal-naive floor** in its GOAT gate. The floor is `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` (Plan 340 with `m=1`, which degenerates to plain split conformal prediction over 1-step residuals). The primitive must beat the floor on CRPS / empirical coverage / Winkler interval score (whichever applies). If it cannot beat the floor, it is not a UQ primitive — it is noise, and the GOAT gate FAILS.

**Retroactive application (existing UQ-bearing primitives; per-policy retroactive audit COMPLETE 2026-06-30, consolidated in `.benchmarks/010_report_the_floor_consolidated.md`):** BoMSampler (Plan 281), Sleep-Time Query Anticipator (Plan 334), Best-Belief Beta Selector (Plan 336), and KARC+conformal-overlay (Plan 308+340) were grandfathered at their current promotion state and have all passed their floor comparison. Future UQ-bearing primitives MUST include the floor from initial GOAT gate (no grandfathering). The floor shipped in Plan 340 Phase 1 (2026-06-30); the rule is now **enforceable**.

**Why:** Manokhin's companion paper *Report the Floor* (arXiv:2606.09473) proves that a trivial training-free conformal interval is a mandatory baseline — any probabilistic forecaster that can't beat it is not adding value over the floor. Adopting this as policy prevents future UQ claims that are actually just the floor in disguise.

### 3. Implement to unblock

If a plan is blocked by a missing primitive, implement the minimal version. After GOAT check + proof of gain: promote to default if it wins, demote the loser.

### 3.5. Modelless unblock protocol — MANDATORY before any riir-train deferral

**Hard rule:** before deferring ANY GOAT gate, plan task, or mechanism to riir-train ("this needs training"), you MUST exhaust all modelless correction paths first. A gate that *appears* to need training may be passable modellessly via training-target decomposition, freeze/thaw, raw/lora hot-swap, or latent-space correction. Deferring to riir-train without checking is the failure mode this protocol prevents.

**Path 0 — training-target decomposition (the Flow Sampling lesson, arxiv 2605.03984):** Before checking paths 1–3, ask: "is the mechanism's value the **training-loop innovation itself** (new optimizer/loss/curriculum/RL algorithm), or the **math the training computes** (closed-form drift, conditional score, Riemannian correction, steering formula, regression target)?" If the latter, decompose the math into components and grep the codebase for the modelless analog of EACH component (see §1 step 2 standing training-math vocabulary). If ALL components have modelless analogs → the mechanism is MODELLESS-VALIDABLE as a fusion of existing primitives — no riir-train deferral needed. **Canonical: Flow Sampling** — the conditional drift formula decomposes into dllm interpolant + Latent Field Steering (Plan 309) reward-gradient + freeze/thaw replay buffer; the training loop is just one way to compute that math. Only if the math has no modelless decomposition → proceed to paths 1–3 below.

**The three modelless unblock paths (check ALL before deferring, AFTER path 0 decomposition fails):**

1. **Freeze/thaw snapshot correction** (`riir-neuron-db/src/freeze.rs`, `MerkleFrozenEnvelope`) — can a frozen snapshot state, thawed at inference, fix the issue? If the failure is a systematic bias from a runtime construction (e.g., doubled signal, position mismatch, attention pattern asymmetry), a corrected snapshot + thaw may eliminate it without gradient descent.
2. **Raw/lora reader-writer hot-swap** (`LoraPair { reader, writer }`, Plan 025; `LoRAHotSwap`, `dispatch_lora_merge` in riir-ai) — can a **deterministically constructed** (not trained) reader or writer adapter fix the issue? Applying a pre-constructed LoRA overlay is modelless (weight addition, no backprop). The question is: can the correction be derived in closed form (e.g., scale-by-0.5, zero-out-specific-positions, identity-minus-projection) rather than learned via gradient descent?
3. **Latent-space correction** (dot-product projection + sigmoid gate, per constraint #2) — can the bias be corrected by projecting the latent state onto a correction direction and gating the output? This is the modelless analog of a trained adapter: instead of learning the correction, derive it analytically from the failure mode.

**Decision protocol:**

```
Gate/mechanism/paper appears to need training
  → Path 0: Is the value the MATH (closed-form drift, conditional score, Riemannian correction, regression target) rather than the training loop itself?
    YES → Decompose the math into components; grep for modelless analog of each (§1 step 2 training-math vocabulary).
      ALL components have modelless analogs → MODELLESS-VALIDABLE as fusion. No riir-train deferral.
      SOME components have no analog → proceed to paths 1–3 for those components.
    NO (value is optimizer/loss/curriculum/RL algorithm) → genuine riir-train candidate; still run paths 1–3 below.
  → Does the failure have a SYSTEMATIC, characterizable cause (e.g., "signal doubled", "position offset", "attention asymmetry")?
    NO → genuine riir-train dependency. Note "→ riir-train", stop.
    YES → Can freeze/thaw (path 1) fix it? Check the freeze envelope API.
      NO → Can a deterministically constructed reader/writer LoRA (path 2) fix it? Check the LoraPair API.
        NO → Can a latent-space projection/gate (path 3) fix it?
          NO → genuine riir-train dependency. Note "→ riir-train", stop.
          YES → implement the latent correction modellessly. Gate is MODELLESS-VALIDABLE.
        YES → construct the LoRA correction modellessly. Gate is MODELLESS-VALIDABLE.
      YES → freeze the corrected state. Gate is MODELLESS-VALIDABLE.
    
  → MODELLESS-VALIDABLE gates must be implemented and checked BEFORE any riir-train deferral.
  → If path 0 + paths 1–3 all fail, THEN note "→ riir-train" with explicit documentation of WHY each path failed.
```

**Documentation requirement:** every "→ riir-train" deferral MUST include:
- **Path 0:** what math components were decomposed, which had modelless analogs, which did not.
- Which of paths 1–3 were checked.
- Why each failed (concrete reason, not "doesn't apply").
- What specifically requires gradient descent that no deterministic construction can provide.

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
- Training-only redirects (→ riir-train, no parity claim).
- Latency-only claims (a single criterion bench suffices, no full PoC).
- Low-confident verdicts that explicitly mark the quality claim as unproven and create a `.issues/` entry to track the PoC follow-up.

**Where the PoC lives:** `riir-ai/crates/riir-poc/` — the "defend-wrong" R&D crate. It exists for exactly this: empirical settlements of disputed primitives before any verdict becomes a feature flag. A PoC has three competitors minimum: the paper's mechanism (or its distilled modelless analog), a frozen/no-adaptation baseline, and the shipped runtime analog. Run them head-to-head on a controlled toy domain (no training), print a verdict table. Use `CARGO_TARGET_DIR=/tmp/...` per the AGENTS.md rule and clean up when done.

**The PoC's job is to defend OR refute.** A PoC that only confirms the verdict is weaker than one that honestly refutes part of it. If the PoC refutes the quality claim:
1. **Do NOT silently revise the verdict to match the PoC.** Record the raw numbers in the research note as a §"PoC Addendum" section.
2. **Honest revision:** explicitly state which claim type was confirmed (architectural, latency) and which was refuted (quality). The verdict stands on the confirmed axes; the refuted axis becomes a tracked follow-up (issue in `.issues/`).
3. **The PoC stays as a permanent regression check** in `riir-poc` — its job was to settle the dispute, and it should keep settling it if the shipped primitive is later tuned.

**Canonical example (Research 360, AdaJEPA, 2026-07-01):** the verdict claimed "parity" between the shipped `ReestimationScheduler` and AdaJEPA's per-MPC-step GD loop, based on architectural coverage alone. The PoC at `riir-ai/crates/riir-poc/benches/adajepa_modelless_goat.rs` confirmed latency parity (~940 ns/replan) and architectural coverage, but **refuted quality parity** — the coherence trigger was too conservative for mild shifts (0 updates at a mild shift), and all adaptation strategies diverged on overshoot shifts. The verdict was honestly revised in a §9 PoC Addendum; the follow-ups are tracked in `riir-ai/.issues/363`. This is the canonical "architectural coverage ≠ quality parity" lesson — grep proved the loop existed, the PoC proved it didn't perform.

### 4. Search if curious

Use web search mcp for that.

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

**Other reference docs:**
- `katgpt-rs/.contexts/optimization.md` — perf rules (zero-alloc, SIMD, rayon, caching)
- `katgpt-rs/.contexts/ibraheemdev-papaya-v0.2.3-examples.md` — papaya lock-free hashmap usage
- `katgpt-rs/.research/004_LoRA_Architecture_Verdict.md` — LoRA / validator terminology
- `katgpt-rs/.research/005_Artifact_Definition.md` — artifact terminology
- `katgpt-rs/.research/238_LoRA_Muon_Spectral_Low_Rank_Manifold.md` — canonical research-note example
- `katgpt-rs/.plans/271_attention_matching_compaction.md` — canonical plan example
- `riir-chain/AGENTS.md` — repo-local context for the chain repo
- `riir-neuron-db/AGENTS.md` — repo-local context for the neuron-db repo
- `katgpt-rs/.research/219_Topological_Neural_Operators_DEC_Inference.md` — DEC operators distillation (the parent note that shipped the Stokes substrate)
- `katgpt-rs/.research/271_MIT_6S184_Diffusion_Flow_Textbook_Vocabulary_Crosswalk.md` — diffusion/flow vocabulary crosswalk (also flags Fokker-Planck as a known gap, closed by Research 296)
- `katgpt-rs/.research/296_Stokes_Calculus_Dec_Vocabulary_Crosswalk.md` — Stokes/Divergence/Hodge vocabulary crosswalk + GOAT verdict for the three wrapper primitives
- `katgpt-rs/.plans/251_dec_operators_cell_complex.md` — DEC operators (COMPLETE — ships `d`, `δ`, `Δ`, `hodge_decompose`, `DecFlowField`)
- `katgpt-rs/.plans/314_stokes_calculus_wrappers.md` — Stokes-calculus wrapper primitives (`belief_mass_divergence`, `boundary_flux_mass`, `line_integral`)

## TL;DR

**Pre-flight (mandatory, before any verdict):** `read_file` 4 READMEs (`katgpt-rs`, `riir-ai`, `riir-chain`, `riir-neuron-db`) + `riir-ai/.docs/README.md` (read `03_pillars/README.md` + `04_supergoat_candidates/README.md` before any Super-GOAT gate); `list_directory` all 4 `.research/` folders + 4 runtime src trees + 7 Super-GOAT factory modules (§Primary focus).

**Workflow:** read paper → classify (training-only? → run §3.5 path 0 training-target decomposition FIRST — if the math decomposes into modelless primitives, NOT riir-train; only if optimizer/loss/curriculum/RL algorithm itself is the value → riir-train, stop) → distill + fuse (vocabulary-translate BOTH semantic names AND compute units per the standing blocks below, then grep BOTH layers — notes+plans+docs AND src+crates — across all 7 repos, using BOTH paper vocab AND codebase vocab) → **latent-space reframe before verdict** (adapter routing / KV compression / speculative decode are GOAT-tier fallbacks, NOT primary) → **§1.55 value-extraction scan (mandatory even on Pass)** → novelty gate (Q1–Q4, §1.5) → MOAT gate per domain (§1.6) → plan + GOAT gate.

**Hard rules:** modelless-first (translate compute units — LLM-as-implementation ≠ LLM-as-mechanism; when you see "N LLM calls/step", ask "what decision is each call computing?" first, not "violates 20Hz budget, NO-GAIN"); latent-to-latent with sigmoid (never softmax); freeze/thaw over fine-tuning; 7-repo discipline; raw scalars at sync boundary; fusion-first mindset.

**Failure-mode prophylactics:** vocabulary translation blocks below (semantic + compute-unit + DEC/Stokes + per-NPC runtime + **substrate-translation for hardware papers** + **database-substrate translation for database/systems/agent-memory-infrastructure papers** + **training-math translation for training/diffusion/flow-matching/sampling papers**); read-the-hits rule (grep hit touching per-NPC+memory+personality+swap → `read_file` TL;DR before claiming novelty); 7 Super-GOAT factory modules; R169 false-trigger guard (decision-structure ≠ LLM-dependent process); **R418 hardware-paper guard (substrate ≠ value — hardware / NMP / ASIC / PIM papers MUST translate the technique to software SIMD before PASS; grep Research 110 Ciot Plasma/Cold tier explicitly; prediction ≠ ceiling — do not anchor perf ceilings pessimistically, mark them as pending the GOAT gate)**; **R300 database-paper guard (database-engine-as-substrate ≠ access-pattern-as-mechanism — database / systems / agent-memory-infrastructure papers MUST translate the access pattern to `riir-neuron-db`'s substrate [Pod + ShardIndex + MerkleFrozenEnvelope + ItemEmbedIndex + vibe KG + Raven/δ-Mem + MAPE-K] before PASS; "we don't have a SQL planner" ≠ "we don't have a database"; grep `riir-neuron-db/src/` explicitly; prefer latent-first [cosine + dot-projection] + neuro-symbolic [KG triples] over external-escalation [`AnyRAG` is optional + preprocess-only])**; **Flow Sampling training-paper guard (training-loop-as-substrate ≠ training-target-math-as-mechanism — training / diffusion / flow-matching / score-matching / sampling papers MUST decompose the training-target math into components and grep for the modelless analog of EACH [dllm interpolant, Latent Field Steering Plan 309 reward-gradient, freeze/thaw replay buffer (NOTE: Spherical Geodesic Steering Plan 405 ships Slerp-the-geodesic only, NOT the Riemannian parallel-transport/Jacobian machinery -- the latter is for sampling-from-distributions-on-the-sphere, which we do not need since our vMF is a deterministic Eq 17 sigmoid gate), induced CWM direction mining] before → riir-train; "this trains a network" ≠ "this is training-only"; PTRM Research 049 cautionary flag on gradient-guided sampling — cite when the fusion touches Langevin)**. **Parity / "already ships" quality claims need a defend-wrong PoC in `riir-ai/crates/riir-poc/` (§3.6) — architectural coverage ≠ quality parity.** **PASS = no files, Gain = files (§1.55). No middle tier. If the paper has actionable improvements, it's Gain. If it only "validates our design", it's Pass.**
