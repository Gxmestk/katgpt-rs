# Massive Activations in Hybrid Linear Attention LLMs — PAS/ISP and the Sink-Position Quantization Prior

**Paper:** Massive Activations in Hybrid Linear Attention Large Language Models: Pre-Attention Spikes and Inter-Spike Plateaus — Zunhai Su, Bohan Sun, Xialie Zhuang, Shuibai Zhang, He Xiao, Jing Xiong, Hengyuan Zhang, Zhongzhu Zhou, Tiantian Zhang, Ngai Wong, Chuan-Wei Kuo [arXiv:2608.12149 "Massive Activations in Hybrid Linear Attention Large Language Models: Pre-Attention Spikes and Inter-Spike Plateaus", Aug 2026, under review]. Code: github.com/StartluxLabs/Massive-Activations-HLA; checkpoints on HF (startlux-models).

**Status:** DISTILLED — pending owner decision. Falsifiable POC filed as `riir-ai/.issues/716_sink_position_aware_q8kv_scale_policy.md`.

**TL;DR:** First systematic study of massive activations (MAs) in layer-interleaved hybrid linear attention (HLA) LLMs. Two architecture-aligned morphologies: MAs spike immediately **before** full-attention layers (pre-attention spikes, PAS) and persist through intervening linear layers when full attention is dense (inter-spike plateaus, ISP); at the full-attention limit this recovers the stable persistent-MA morphology of ordinary Transformers. Mechanism: a shared **write–sink–cancel** lifecycle distinguished only by cancellation timing. For this stack: (1) it confirms our sigmoid-attention default removes the MA *cause* (softmax) rather than the symptom; (2) it exposes a live gap in the Q8KV substrate (per-block absmax, zero outlier/sink awareness — the paper's exact shared-scale failure mode); (3) it adds a cheap, previously unused quantization axis — **hybrid layer-schedule position + sink-token position** — for the Kimi-K3-class hybrid models we run.

## What the paper establishes

1. **PAS/ISP phenomenology (new).** Across 5 linear-attention backbones (RetNet, HGRN, GLA, DeltaNet, GDN) × 6 hybridization ratios × 5 domains × models 1.2B→397B (Kimi Linear 48B, Qwen3.5 35B/122B/397B, Nemotron-H 8/47/56B, Zamba2 1.2/2.7/7B): the sink token's max-|hidden-state| spikes at the layer immediately preceding every full-attention layer (sink–spike alignment ≈100%, Table 1). Sparse full attention → isolated spikes; dense (3:1) → sustained inter-spike plateaus (ISR rises monotonically with density, e.g. GDN 18%→27%→78%); ρ=1 → the conventional stable MA plateau.
2. **Magnitude–sink alignment breaks under hybridization.** In HLA models, magnitude ranking alone is an unstable MA tracker (max-token switching up to 62% between adjacent layers); the paper introduces attention-derived **consensus sinks** (Eq. 5: attention mass averaged over full-attention layers/heads, normalized by valid-query count) as stable anchors.
3. **Write–sink–cancel lifecycle (new).** Fixed-coordinate evidence: the pre-attention layer writes an extreme outlier into the residual stream at (sink token, feature j⋆); during full attention that token becomes the attention sink; a subsequent opposite-signed update at the *same coordinate* cancels it. PAS = prompt cancellation; ISP = delayed cancellation; the full-attention stable morphology = the persistence limit. One mechanism, three timing regimes.
4. **Gating asymmetry (consolidates Qiu et al. 2505.06708).** Element-wise output gating on the (few) full-attention layers attenuates PAS/ISP magnitude far more than removing all GDN output gates — but never eliminates the layerwise organization (Qwen3.5, natively output-gated, still shows both).
5. **Emergence.** Controlled pretraining (340M/1.3B GDN hybrids, 10B/50B tokens): PAS visible after 1B tokens, consolidating thereafter; placement depth sets PAS magnitude (layer 4 weak → layer 20 strongest); downstream retrieval (NIAH/FDA/SWDE) strongly depends on full-attention placement even at ≈100% alignment.

## Prior-art check (novelty gate Q1 — partial)

| Claim | Verdict | Prior art |
|---|---|---|
| PAS/ISP in HLA models | **New** — no prior work documents activation outliers in interleaved hybrids; the M-A-P suite analysis (2507.06457) explicitly did not cover activations | — |
| Write–sink–cancel + cancellation-timing account | **New** as a unified lifecycle | Sun et al. 2402.17762 (MA existence, bias role); "Understanding Massive Values" (ICML'26, layer origin) |
| Gating attenuates MAs, sinks persist | Consolidating | Qiu et al. 2505.06708 GatNAC; Softpick 2504.20966 |
| MAs dominate shared quant scales (KV/activation) | Consolidating | KIVI 2402.02750, RotateKV 2501.16383, KVSink 2508.04257, OSCAR — all full-attention models; this paper extends to hybrid caches |
| Sink pressure exists in hybrids | Adjacent | Hymba 2411.13676 (engineered meta-tokens to absorb sink load — design fix, not analysis) |

## Distillation — what transfers to this stack

### 1. Architecture confirmation (zero work, moat line)

Our default attention (`parallax_attn`/`funcattn`, sigmoid — Research 140/257/261) is **sink-free by construction**: no softmax ⇒ no attention sink ⇒ no MA lifecycle. The paper shows the MA morphology is *organized by softmax-attention placement* and that gating — the strongest known mitigation — only attenuates it. We remove the cause, not the symptom. One-line positioning: *softmax hybrids farm their quantization enemy at predictable layer boundaries; sigmoid attention never breeds it.* (Our shipped sink handling — PFlash sink-block retention, ShardKV lossless sink+window — covers consuming softmax models, which is where sinks are needed.)

### 2. Live substrate gap: Q8KV shared-scale blindness (actionable → Issue 716)

`riir_engine::quant::q8kv` + `attention_q8kv_cubecl.rs` (Gemma 2 2B decode today) quantizes K/V to Q8_0 with per-32-block absmax scales per (position, head). MA geometry: MAs are sparse *channels* (1-2 features at 100-300× typical) on *sink tokens* (position 0, delimiters) that attract disproportionate attention mass. Per-block absmax does not see either prior:
- Intra-block: one 300-magnitude channel sets the block scale; the row's ~1-10 channels drop to ~1 quant step (the KV-side twin of Research 085/086's weight-side outlier→scale-collapse).
- Output: sink rows receive large attention mass ⇒ V-row quantization error goes straight into the output.

Mitigation candidates (A/B in the issue): (a) sink-position lossless sidecar — keep the first S KV rows in f16 (S×heads×head_dim×2B ≈ KBs; decode-time sinks are the oldest, fixed cache positions); (b) KVarN dual-scale (Hadamard + Sinkhorn variance-norm — our shipped outlier suppressor, Research 159) applied per-row-group; (c) channel split (deferred in Research 020 TurboQuant as production-scale-only).

### 3. New quantization axis: hybrid layer-schedule position (forward-looking)

For Kimi-K3-class hybrids (`kimi_k3` decoder: KDA everywhere + MLA every 4th layer, `full_attn_layers` schedule) the KV cache exists **only** at the sparse full-attention layers, and PAS/ISP say the MA-carrying positions localize at exactly those boundaries + sink tokens. Two priors nobody in our stack uses:
- **Positional**: sink entries are known a priori (first tokens) — protect losslessly, near-free because the hybrid KV cache is small.
- **Schedule**: pre-MLA layers are enumerable from the config — any future activation-quantization near those layers should carry per-boundary scale budgets.

### Fusion

Paper × Research 258 (sink dual mechanism NOP/Broadcast — names the massive-activation sink mechanism) × Research 159 (KVarN dual-scale outlier suppression) ⇒ **schedule- and position-conditioned KV quantization policy**: quantization parameters become a function of (layer role in hybrid schedule, token sink-ness) instead of uniform per-tile/per-block. No cousin alone has the positional or schedule axis; the paper supplies the empirical law that makes both priors cheap (MAs are exactly where those two axes intersect). Secondary fusion: Plan 306 `depth_invariance` (`magnitude_slope` on hidden-state chains) is the existing tool to *measure* PAS-like spikes on models we run — the natural verifier for any schedule-aware policy.

### Latent-raw / game-context reframing (steps 3-4, recorded for the audit trail)

Latent reframe: write–sink–cancel is a recognizable dynamic on our per-NPC belief substrate (`sense::evolve_belief`) — observation writes a large update, a broadcast op (CLR weighted set attention, zone attention) distributes it, decay cancels it; cancellation *timing* (prompt vs delayed) maps to salience-persistence tuning. Honest verdict: an analogy, not a primitive — nothing in the game loop consumes LLM hidden-state MAs, and our belief kernels already bound magnitudes by construction (Research 151 hygiene). No game-context behavior class emerges (step 4: no per-NPC scalar the paper's guarantee bounds that we don't already bound). This is why the verdict is GOAT-tier engineering gain, not Super-GOAT.

## What does NOT transfer

- Their controlled pretraining recipe (340M/1.3B GDN hybrids on FineWeb-Edu) — analysis apparatus, not a training-method contribution. No riir-train plan.
- Consensus-sink tracking (Eq. 5) — requires full attention matrices at inference; a diagnostic for model analysis, not a decode-path primitive for us.
- Gating interventions — architecture changes to *their* models; our models are already sigmoid-gated or consumed as-is.

## Verdict: **Gain (GOAT-tier)**

- Q1 prior art: partial (sink-aware KV quant is a populated field — KIVI/RotateKV/KVSink/OSCAR; the hybrid-schedule axis is unclaimed but incremental).
- Q2 new behavior class: no (diagnostic lens + engineering policy).
- Q3 selling point: no ("layer-schedule-aware KV quant" is an internal quality axis).
- Q4 force multiplier: yes-ish (connects KVarN, q8kv, kimi kernels, sigmoid-attention moat) — but Q1-Q3 cap it at Gain.

Actionable because: (a) it exposes a failure mode with no existing mitigation on the *live* q8kv path (reverse-grep: Research 020 explicitly deferred outlier channel splitting; Research 159 KVarN exists but is not wired to q8kv; nothing is sink- or schedule-aware); (b) it supplies the empirical law (sink tokens + pre-attention boundaries = MA carriers) that makes the mitigation cheap. Falsifiable POC: `riir-ai/.issues/716`.

## Closest cousins in-repo

- `katgpt-rs/.research/258_Attention_Sink_Dual_Mechanism_NOP_Broadcast.md` — sink mechanism taxonomy; names ‖x_s‖ massive activation.
- `katgpt-rs/.research/159_KVarN_Variance_Normalized_KV_Cache_Quantization.md` — shipped outlier-suppressing dual-scale KV quant (the mitigation substrate).
- `riir-ai/.research/085_Quantization_Outlier_Collapse…` + `riir-train/.research/086` — outlier→shared-scale collapse (weight-side twin).
- `riir-ai/.research/329_kimi_linear_kda_distillation.md` — KDA output gate ↔ attention-sink coupling in our hybrid substrate.
- `katgpt-rs/.research/020_TurboQuant…` — deferred outlier channel splitting (the documented gap this paper pressures).
- `katgpt-rs/.research/151_Recursive_Latent_State_Magnitude_Hygiene…`, Plan 306 `depth_invariance` — magnitude tooling.
