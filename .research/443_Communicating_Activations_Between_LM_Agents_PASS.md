# Research 443: Communicating Activations Between Language Model Agents (PASS — already shipped)

> **Source:** [Communicating Activations Between Language Model Agents](https://arxiv.org/pdf/2501.14082) — Ramesh & Li (Kempner Institute, Harvard), ICML 2025
> **Date:** 2026-07-16
> **Status:** Done — PASS-with-gains
> **Classification:** Public (katgpt-rs/MIT)
> **Related Research:** 247 (CS-KV Probe / NPC Mind-Reading — the Super-GOAT that subsumes this), 167 (riir-ai Crowd Joint Inference), 289 (RecursiveMAS PASS — identical verdict pattern), 357 (Neural Procedural Memory Activation Steering), 402 (Latent Bridge Slow-Fast — identical "cross-model projection already ships" finding)
> **Related Plans:** 311 (Mind-Reading runtime), 355 (Crowd Attention runtime), 391 (Latent Steering Bridge)
> **Verdict: PASS-with-gains.** The activation-communication mechanism is a **strict subset** of what already ships across three runtime modules. Cherry-pickable design insights recorded in §2.2.

---

## TL;DR

The paper proposes inter-LLM communication via intermediate-layer activation grafting: pause LM B at layer j, replace its last-token activation with `f(hA,k, hB,j)` for simple f (sum/mean/replace) or a task-agnostic learned W projection, continue B's forward pass. Reports up to 27% improvement over natural-language debate at <1/4 the compute on coordination games + reasoning benchmarks.

**This mechanism already ships at strictly higher fidelity** across three modules in our quintet:

| Shipped module | What it covers vs the paper |
|---|---|
| `riir-ai/crates/riir-engine/src/latent_steering_bridge.rs` (Plan 391) | **The literal mechanism**: graft a latent vector into a local-weight LLM's residual stream at a specified `layer` with a game-rules-weighted gate. `ResidualField { layer, steering, weight }` + `forward_with_steering` IS activation communication — the "other agent" is the game substrate (HLA scattered to n_embd), not another LLM. `apply_latent_steering_weighted` is the `f` function. Our version adds the fog-of-war anti-omniscience gate the paper does not have. |
| `riir-ai/crates/riir-engine/src/crowd_attention.rs` (Plan 355, R167) | **The multi-agent generalization**: per-tick, per-zone, sigmoid-gated set attention over visible peers' HLA beliefs with frozen Q/K/V projections. Strictly more expressive than the paper's single-graft f (attention with learned weights > simple replace/sum/mean). |
| NPC Mind-Reading bus (R247/R133 → P311/P280) | **The Super-GOAT**: adaptive-bandwidth latent communication with a fog-of-war context-awareness axis (~25× density swing). The paper has no analog of this axis. |

The verdict pattern is **identical to Research 289 (RecursiveMAS PASS)**: "Every modelless primitive is already shipped at higher fidelity. The cross-agent latent comms selling point is already Super-GOAT (R247/R133 → P311/280) with the fog-of-war adaptive-bandwidth axis." Paper 2501.14082 is *simpler* than RecursiveMAS (single-step graft vs recursive loop), so if RecursiveMAS was PASS, this is definitively PASS.

---

## 1. Paper Core Findings

### 1.1 The mechanism (§3.1)

Given LMs A and B:
1. Run partial forward of B to layer j → `hB,j ∈ R^{tB×dB}`
2. Run partial forward of A to layer k → `hA,k ∈ R^{tA×dA}`
3. Replace B's last-token activation: `(hB,j)_tB ← f((hA,k)_tA, (hB,j)_tB)`
4. Continue B's forward pass to decoding

Three non-learned f: `sum (a+b)`, `mean (a+b)/2`, `replace (a)`. Optional task-agnostic linear W (`R^{dB×dA}`) trained once per model pair via MSE on general text (C4), projecting A's space onto B's.

### 1.2 Why it works (§3.3)

Citing Hernandez et al. (2024) + Hewitt & Manning (2019): by ~50% depth, an LM has developed "enriched entity representations" — linear probes predict output characteristics extremely well. By the final layers, the embedding is transformed toward next-token prediction, "throwing away" information not useful for decoding but useful for communication. **The intermediate layer is richer than the output.** This is why activation graft at layer k communicates more than the decoded token.

### 1.3 Results

- **Coordination games** (Countries, Tip Sheets): AC with f=replace recovers ~91-95% of the gap between silent and skyline, beating NL communication at far less compute.
- **Reasoning benchmarks** (7 datasets): AC beats both single-model baselines AND NL debate (up to +27pp) on 6/7 datasets, using <1/4 the compute. Cross-family (Qwen/Gemma/LLaMA) works without W.
- **f=replace > mean > sum**: replace guarantees output stays in B's activation space; sum doubles the norm (out-of-distribution).
- **Last-token only suffices**: masked attention already aggregates context into the last-token activation. Modifying all tokens hurts (Table 5).
- **Single-step suffices**: no rounds needed (unlike NL debate). One graft communicates "all of A's knowledge."
- **Activation-space similarity ∝ AC gain** (Appendix B.4): matrix-cosine-similarity of A/B activation spaces correlates with performance gain.
- **In-distribution W helps** (Appendix B.3): GSM8k 64% → 78% when W trained on GSM8k train set vs C4.

---

## 2. Distillation

### 2.1 What already ships (the PASS grounding)

| Paper component | Shipped cousin | Plan / Research | Higher-fidelity aspect |
|---|---|---|---|
| Activation graft at layer j into LLM B's residual | `latent_steering_bridge.rs::forward_with_steering` + `ResidualField { layer, steering, weight }` | Plan 391 | Adds game-rules visibility gate (anti-omniscience); multi-field multi-layer; source is game substrate (HLA scatter), not another LLM |
| Combine function f (sum/mean/replace/learned-W) | `apply_latent_steering_weighted` (weighted add) + `crowd_attention` residual consensus `h_i + γ·Σ α_ij·(v_j − h_i)` | katgpt-core `latent_steering`, Plan 355 | Crowd attention uses Q/K/V attention (strictly more expressive than simple graft); steering uses weighted addition with fog-of-war gate |
| Cross-agent latent communication (the selling point) | NPC Mind-Reading adaptive-bandwidth bus | **R247, R133 → P311, P280** | Adds fog-of-war context-awareness axis (~25× density swing); CS-probe ranks which dims carry signal |
| Task-agnostic mapping matrix W (MSE on C4) | `substrate_direction` adapter (scatter HLA 8-dim → LLM 3072-dim via pre-computed channel maps) | Plan 391 | **Modelless** — no training needed, deterministic scatter |
| Multi-agent set communication | `crowd_attention.rs` cross-NPC set attention | Plan 355, R167 | Permutation-equivariant; sigmoid-gated (not softmax); frozen Q/K/V from CS-rankings + functor directions |
| Intermediate-layer richness insight | HLA IS the "intermediate enriched representation"; `depth_invariance_audit.rs` (R286/P306) | R242, R286 | HLA is a per-NPC recurrent belief kernel at the affective layer — exactly the "halfway-point enriched representation" |

**This is the same prior-art surface Research 289 (RecursiveMAS) already mapped.** RecursiveMAS was PASS because every modelless primitive shipped at higher fidelity; the cross-agent latent comms was already Super-GOAT (R247/R133/P311). Paper 2501.14082 is a single-step simplification of the same class — it adds nothing the shipped system doesn't already cover.

### 2.2 Cherry-pickable gains (§1.55 value-extraction scan)

Six questions, four "yes":

1. **Config tuning?** The paper's "replace > sum > mean" does **NOT** transfer to our heterogeneous case. Replace works for the paper because A and B are similar LMs with aligned activation spaces. In our case: (a) `crowd_attention` consensus update preserves NPC individuality — replace would homogenize the crowd; (b) `latent_steering_bridge` grafts a game-derived vector (not another LLM's activation) — replace would destroy LLM context. **Honest negative result: the combine-function ranking is homogeneous-LLM-specific and does not transfer.**

2. **Design principle?** **YES.** The intermediate-layer richness finding (§3.3, citing Hernandez 2024) confirms our architecture: HLA IS the "halfway-point enriched entity representation" analog — a per-NPC recurrent belief state that is richer than any decoded token. For the LLM steering bridge, the `ResidualField.layer` field should default to ~50% depth (the paper fixes k=j=26 out of 32 layers ≈ 81% — deeper than 50%, suggesting the richness peak for factual recall is past halfway). Record as a design principle for the `layer` default.

3. **Composition schedule?** **YES.** Single-step sufficiency (no rounds needed) validates the per-tick single-step `crowd_attention` design. One attention pass per tick communicates all peers' beliefs — no intra-tick iterative refinement needed. This matches the paper's finding that one graft communicates "all of A's knowledge."

4. **Empirical validation target?** **YES.** The paper's compute formula (§3.2) and the ~4× compute savings vs NL debate is an evidence template for our claim that latent-to-latent beats text round-trips. `latent_steering_bridge.rs` already asserts "no text round-trip, no API, no tokenization" — this paper quantifies the win.

5. **Failure-mode exposure?** **YES.** Activation-space similarity ∝ AC gain (Appendix B.4) is a validation target for cross-class NPC communication (Knight HLA vs Mage HLA). Same-class NPCs (homogeneous 8-dim HLA) need no projection; heterogeneous classes need the W / scatter projection. Connects to R247's "cross-shape projection needs training" caveat (currently P3, unblocked).

6. **Benchmark domain insight?** The coordination games (Lewis signaling games: Countries, Tip Sheets) map to our NPC coordination scenarios but don't reveal a regime we haven't already targeted.

### 2.3 Latent vs raw boundary

No new boundary crossing. The paper's mechanism operates entirely in latent space (intermediate activations), consistent with our latent-to-latent preference. The shipped versions already respect the sync boundary: `crowd_attention` output stays per-NPC latent; `latent_steering_bridge` operates on the local LLM residual; only the existing 5 emotion scalars cross sync.

---

## 3. Verdict

**PASS-with-gains.** The activation-communication mechanism is a strict subset of what ships across `latent_steering_bridge.rs` (LLM residual grafting), `crowd_attention.rs` (cross-NPC set attention), and the NPC Mind-Reading bus (R247/R133/P311 — adaptive-bandwidth latent comms with fog-of-war context-awareness). The verdict pattern is identical to Research 289 (RecursiveMAS PASS): every modelless primitive is already shipped at higher fidelity; the cross-agent latent comms selling point is already Super-GOAT with the fog-of-war axis this paper lacks.

| Gate | Criterion | Honest answer |
|---|---|---|
| **Q1** No prior art? | **FAIL.** Activation graft at intermediate layer ships as `latent_steering_bridge.rs`. Cross-agent latent comms ships as R247/R133/P311. Multi-agent set attention ships as `crowd_attention.rs` (R167/P355). |
| **Q2** New class of behavior? | **FAIL.** "Inter-agent activation communication" IS the Plan 311 selling point — and our version adds the fog-of-war adaptive-bandwidth axis. |
| **Q3** Selling point? | **FAIL for new selling point.** Already Super-GOAT (R247). |
| **Q4** Force multiplier? | **YES but only as a redescription** of the R247/R167 connection map. |

**Routing:** No new primitive, no plan, no guide, no `.issues/` entry. The cherry-pickable gains are design insights recorded in §2.2 for future reference. No config change warranted (the "replace > sum > mean" finding is a honest negative result for our heterogeneous case — see §2.2 gain #1).

**Honest caveat (§3.6 compliance):** this PASS is backed by architectural coverage (grep + read of `latent_steering_bridge.rs`, `crowd_attention.rs`, R247/R167/R289). No quality-parity claim is made — the claim is "the mechanism ships," not "the shipped version matches the paper's numbers." A head-to-head PoC is not required because no quality parity is asserted; the PASS is an architectural redirect, not a "our version works as well" claim.

---

## TL;DR

PASS-with-gains. The paper proposes inter-LLM activation grafting at intermediate layers (f = sum/mean/replace/learned-W). This is a **strict subset** of three shipped modules: `latent_steering_bridge.rs` (graft game-derived latent into LLM residual at layer j — the literal mechanism, Plan 391), `crowd_attention.rs` (cross-NPC set attention on HLA — strictly more expressive, R167/P355), and the NPC Mind-Reading bus (adaptive-bandwidth latent comms with fog-of-war context-awareness — Super-GOAT, R247/R133/P311). Verdict pattern identical to Research 289 (RecursiveMAS PASS): every modelless primitive already ships at higher fidelity. Four cherry-pickable gains: (1) intermediate-layer richness confirms HLA-as-enriched-representation framing + `ResidualField.layer` ~50-80% depth default, (2) single-step sufficiency validates per-tick non-iterative `crowd_attention`, (3) compute-savings formula is evidence template for latent>text claim, (4) activation-space similarity ∝ gain validates cross-class projection need. One honest negative result: "replace > sum > mean" does NOT transfer to our heterogeneous case (game-substrate→LLM, individual-NPC→crowd). No new primitive, no plan, no guide, no issue.
