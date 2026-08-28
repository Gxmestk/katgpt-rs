# Research 490: DFlash 2 — Pair-Scored Path Selection over Parallel Draft Candidates

**Status:** DISTILLED → POC EXECUTED, G2 FAIL (2026-08-19) — [Bench 699](../../riir-ai/.benchmarks/699_issue671_pair_scored_selection_gate.md): the modelless composition (t-step table marginals as `U`) does not lift chain acceptance (0.2855 vs 0.5 gate; chain 0.2845, tree ceiling 0.8770) — the two signals are collinear; G1 PASS, entropy gate = dead weight. Substrate kept opt-in as the negative-result artifact; the headroom redirect is a real-drafter `U` (riir-train lineage) or Issue 717 tree-verify.

> **Source:** "DFlash 2: Keep Drafting Parallel" — Inco AI blog, 2026-08-18, https://inco.ai/blog/dflash2/ (blog-only; no arXiv as of 2026-08-19). Successor to DFlash [arXiv:2602.06036, Z Lab, ICML 2026].
> **Date:** 2026-08-19
> **Status:** Active — distilled; POC filed as Issue 671
> **Related Research:** 316 (DSpark — the semi-AR baseline + the modelless-Markov-head path that shipped as `bigram_markov.rs`), 407 (TfM/Weaver — the acceptance-ceiling analysis + trained-adapter alternative), 177 (Domino — decoupled causal correction), 149 (FlashAR — dual-path consensus), 243 (Bebop acceptance forecast)
> **Related Plans:** Plan 339 (HardwareAwarePrefixScheduler — DSpark distillate)
> **Cross-ref (riir-ai):** Bench 693/694 (Issue 717 consumer gates — the measured selection headroom this note targets), katgpt-rs Issues 659 (bigram head), 670 (TreePath fix, RESOLVED), 671 (this note's POC)
> **Classification:** Public
> **PASS-Redirects (synthesis):** Valluri, Nguyen & Grover [arXiv:2608.20359 "Self-Speculation for Faster Reasoning Models"] — adds a drafter-SOURCE class orthogonal to this note's selection problem (same model at partial CoT budget as its own drafter — intersection-novel vs LayerSkip/QuantSpec/SpecReason per prior-art search) and reports suffix-only verification BEATS prefix-only (1.318× vs 1.086× on ClassEval): span recovery beyond the first rejected token carries more reuse than prefix acceptance in their regime. PASS for us: our closest analog, `fill_lookup_draft` (Issue 742 T2, 15.15/16 acceptance), already saturates the high-overlap regime where suffix decoding pays, and their suffix cache composes with lookup — a bounded ≤5% ceiling here. Vocabulary hazard: this note's "suffix decay" (end-of-block recall collapse in parallel drafters, DFlash 2's two-tap conv fix) is UNRELATED to SSR's "suffix decoding" (suffix-cache span recovery during continuation) — different mechanisms sharing a word.
> **PASS-Redirects (synthesis):** vLLM blog 2026-08-23 ["Exploring Speculative Decoding in vLLM on AMD GPUs"](https://vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus) — external confirmation of this note's suffix-decay economics at serving scale: vLLM's per-position heatmaps show p1 acceptance flat 84–95% across N while tails collapse to <10% by p10–15, and DFlash throughput peaks at N=7 (2.87×) then REGRESSES at the full N=15 block (2.40×) — the same non-monotonic-block shape as our K=16 knee (Issue 742, 15.15/16). Their §4-flip nuance (DSpark beats DFlash on Qwen3-8B but loses on gemma-4-31B, Research 518 §4) strengthens this note's "architecture is a budget, data is the variable" framing: never port a drafter-family ranking across checkpoints. Record: Research 518.

---

## TL;DR

DFlash 2 adds two mechanisms to parallel block drafting: (1) a **path selector** — keep the top-16 candidates at each position from the parallel drafter and score every **adjacent pair** with a low-rank bilinear form `S_t(a,b) = U_t(b) + ⟨A(a)⊙H(h_t), B(b)⟩`, then greedily walk one coherent path through the candidate lattice (+2.0M params, +0.6% cycle latency; beats DSpark's sequential Markov correction with 40× fewer params and 16× lower latency overhead); and (2) a **two-tap dynamic depthwise convolution** fixing **suffix decay** (end-of-block recall collapse) at 3% params where 10 extra layers cost 15.2% latency. Net: +21% acceptance length over DFlash, 2.7–3.4× autoregressive throughput, lossless.

**Distilled for katgpt-rs (modelless, inference-time):** the selector's two scoring signals — per-position parallel-marginal evidence `U_t(b)` and adjacent-pair coherence — **both ship separately** (`dflash.rs` marginals + `dd_tree`; `BigramMarkovTable` transitions) but are **never combined in one path selection**: `extract_best_path` scores nodes by marginal only; the bigram chain conditions on transitions only. The modelless composition `S(a,b) = U(b) + λ·log P_bigram(b|a)` (+ an entropy-sigmoid per-position gate as the `H(h_t)` analog) is the DFlash 2 mechanism with zero training, and it targets the measured Bench 694 headroom directly: greedy chain acceptance 0.2984 vs tree ceiling ≈0.88–0.89 — with chain-verify cost, not tree-verify cost. The suffix-decay-is-local finding is independent validation of the bigram line's core bet (Issue 659).

---

## 1. Core Findings

### 1.1 The selection-headroom decomposition

DFlash drafts every position independently in one pass. Per-position candidate lists already contain the right token: Recall@1 at position 0 is 85.4% but **Recall@16 is 99.5%** (5-layer Qwen3-4B on GSM8K, conditioned on earlier positions correct). An oracle picking the right candidate from top-16 lifts acceptance length **4.27 → 6.79**. The gap is "pure selection headroom" — the candidates are there; the top picks don't fit together (e.g. two neighbors both pick the same word → stutter dies at verification).

This is the same decomposition as Research 407 §1.1's factorized-drafter acceptance ceiling (marginals average over prefixes, diluting signal; conditioning on realized draft tokens within the support beats pure marginals). DFlash 2 is a **second, much cheaper data point** for the same thesis: Weaver (407) restores dependencies with a trained 56.7M autoregressive adapter; DFlash 2 does it with a 2.0M bilinear pair scorer whose scoring stays **fully parallel**.

### 1.2 The path selector (the mechanism that transfers)

Keep top-16 candidates per position. Score every adjacent pair:

```
S_t(a,b) = U_t(b) + ⟨A(a) ⊙ H(h_t), B(b)⟩
```

- `U_t(b)` — the drafter's own logit for `b` at position `t` (context evidence, independent per position)
- `A(a), B(b)` — 256-dim token embeddings; `H(h_t)` — context gate deciding which parts of the match count. "In essence, this is a **low-rank bilinear attention over adjacent candidates**."
- All pairs scored in **one shot** — no extra backbone or LM-head pass. Only the final walk is sequential: greedy best-successor from the last verified token (or sampling from the same scores; rejection sampling restores the exact target distribution — lossless).

| Method | Params | Latency | T=0 | T=1 |
|---|---|---|---|---|
| DFlash | — | — | 4.27 | 3.78 |
| + DSpark correction | +77.8M | +9.6% | 4.49 | 4.08 |
| + path selection (DFlash 2) | **+2.0M** | **+0.6%** | **4.61** | **4.25** |

"**Choosing is cheaper than predicting.**" Selection beats DSpark's sequential rewrite on both temperatures at 40× fewer params — because the correction pass in DSpark/Domino-style heads is inherently left-to-right (serializing), while pair scoring keeps everything parallel.

### 1.3 Suffix decay is a local problem (the conv fix)

Recall decays toward block end even for the oracle (99.5% → 87.8%) — the candidates themselves run out; no selector can fix it. Diagnosis: within-block attention mass collapses 30% (Layer 1) → 8% (Layer 5), concentrating in a few heads. Fix: split the jobs — a **two-tap dynamic depthwise convolution** before/after each attention+FFN sublayer takes the within-block work:

```
Conv_k(x)_t = k_{t,0} ⊙ x_t + k_{t,1} ⊙ x_{t-1}
```

(learned base kernel + content-dependent correction, 16 channels share one correction; position 0 reads the last verified token). +16.5M params (3%), +0.7% latency — recovers most of what 10 extra layers buy (15.2% latency). Within-block attention across Layers 4–5 falls 9.4% → 0.5% (the conv absorbs the local work). Two key diagnoses: "**coherence is mostly local: a candidate's fit depends mainly on the token just before it**" and "**suffix decay is mostly a local problem**."

### 1.4 Combined results

Qwen3.5-4B mean acceptance: MTP 4.54 / DFlash 4.92 / DSpark 5.49 / **DFlash 2 5.97** (+21% over DFlash). Qwen3.8-27B: 4.80 vs MTP 4.28 vs DSpark 3.62 → 2.7–3.4× AR throughput (SGLang, bs 1). Muse Glimmer: 5.70 vs official DFlash 4.44 → 3.1–4.6×. Selector + conv together add 1.3% cycle latency.

### 1.5 The DSpark contract (confirms Research 316; explains Bench 694 P2)

The blog describes DSpark/Domino correction as "sequential Markov heads that **rewrite each position's full-vocabulary distribution**" — i.e. a prefix-dependent transition bias added to base logits, applied left-to-right. This is exactly Research 316 §1.2's documented contract (`B_k` added to `U_k` before softmax, sampled left-to-right) and explains why Bench 694 P2's **standalone** read of DSpark's trained rank-256 head was inconclusive (1.76% top-1 vs the modelless bigram table's 22.27%): the head is a composed bias under log-SNR conditioning, never a standalone predictor. G2-full stays OPEN numerically (needs the DSpark forward), but the contract question is now settled by two independent sources agreeing.

---

## 2. Distillation

### 2.1 What already ships (signal-diff per §3.6 — read the cousin's formula, not its name)

| DFlash 2 component | Shipped cousin | Signal each consumes | Coverage |
|---|---|---|---|
| Parallel per-position candidates `U_t` | `dflash_predict_parallel` (`katgpt-speculative/src/dflash.rs`, shared core w/ riir-engine) | context-conditioned per-position marginals | ✅ ships |
| Candidate lattice + best-path walk | `dd_tree` (`build_dd_tree*`, `extract_best_path_into`) + `best_of_k_rollouts` | per-node marginal scores; walk = per-depth best by node score | ⚠️ **no transition term** |
| Adjacent-pair coherence | `BigramMarkovTable` (`bigram_markov.rs`, Issue 659 / Research 316 §3.5 path 2) | `P(next\|prev)` transitions — full sparse table, greedy chain + tree expansion | ⚠️ **no per-position context evidence** |
| Sequential correction (the thing DFlash 2 beats) | `domino.rs` `PrefixCorrectionTable` + `domino_score` (Plan 197 / Research 177) | prefix-conditioned logit residuals, applied left-to-right | ✅ ships (the baseline shape) |
| Context gate `H(h_t)` | `AcceptanceForecast` (Bebop, Plan 243) — `α ≈ a − b·H(p)` | per-position entropy → acceptance | ✅ shape ships (modelless gate analog) |
| Lossless T>0 | `LeviathanVerifier` RS acceptance = 1−dTV | draft/target distributions | ✅ ships |
| Suffix-decay conv | — (training-side; see §3 redirect) | — | ❌ n/a modellessly |

**The gap (Q1 novelty, signal-diffed):** no shipped mechanism combines per-position parallel evidence `U_t(b)` **and** adjacent-pair transition `log P(b|a)` in one path selection. `extract_best_path` consumes marginal relevance only; the bigram greedy chain consumes transitions only; domino applies residuals sequentially (the expensive shape). DFlash 2's selector is precisely the composition — and in our stack it is **constructible modellessly**: the bigram table replaces the trained `A·H·B` bilinear (a full sparse table where DFlash 2 uses rank-256), and a per-position entropy-sigmoid gate replaces `H(h_t)`.

### 2.2 Why this matters here — the measured headroom (Bench 693/694)

The workspace has already measured DFlash 2's selection-headroom decomposition on its own seam (proxy corpus, Bonsai tokenizer, vocab 248,320):

| arm | acceptance |
|---|---|
| greedy bigram chain (depth 8) | **0.2984** tok/cycle |
| tree acceptance (budget 256) | **0.8940** (pre-TreePath lower bound; 0.8785 exact per 693 addendum) |
| committed/cycle ceiling (1 + best tree) | 1.894 |

Same shape as 4.27 → 6.79: the candidate structure contains the target path ~89% of the time; the chain picker realizes ~30%. And the G3a FAIL economics make selection the right lever: the loss was **verify-side** (K sequential verify forwards + rollback per cycle), so "the economics need ~≥0.5–0.7 [acceptance] to break even at K small" — exactly the band between greedy chain and tree ceiling. A pair-scored selector that lifts chain acceptance toward ~0.5+ makes the **chain seam viable at chain-verify cost**, with no batched tree-verify harness required (the armed Issue 717 re-gate's other path).

### 3.5-path-0 inventory (per component):

| Component | Coverage | Modelless extraction |
|---|---|---|
| Pair score `S(a,b)` | partial (`bigram_markov` transitions; no `U` term) | ✅ closed-form: `U(b) + λ·log P(b\|a)` from shipped parts |
| Candidate lattice | ✅ (`dd_tree` per-depth candidates / top_m) | ✅ already ships |
| Context gate `H(h_t)` | shape ✅ (Bebop entropy) | ✅ per-position sigmoid gate on λ (entropy or forecast-driven) |
| Greedy walk / sampling | ✅ (`extract_best_path_into` walk shape; `LeviathanVerifier` for lossless) | ✅ reuse |
| Trained bilinear `A,B` (rank-256) | ❌ | **not needed** — full bigram table is the deterministic construction (path 2, the Issue 659 precedent) |
| Two-tap dynamic conv | ❌ | ❌ genuinely trained → redirect (§3) |

### 2.3 Fusion

1. **× `bigram_markov` + `dflash.rs` + `dd_tree` (the POC, Issue 671):** score the existing per-position candidate lattice with `S(a,b) = U(b) + λ·log P_bigram(b|a)`; walk greedily (or max-product — O(K·m²) Viterbi over a width-m lattice is trivial); verify the selected chain. Baselines: argmax-of-marginals chain (DFlash-1 style), greedy bigram chain (0.2984), tree ceiling (0.8785–0.8940).
2. **× Bebop `AcceptanceForecast` (Plan 243):** per-position entropy modulates λ — the modelless `H(h_t)`. High-entropy positions trust the pair term more (the marginal is diluted — the 407 ceiling insight); low-entropy positions trust `U`.
3. **× `HardwareAwarePrefixScheduler` (Plan 339 / Research 316):** the scheduler consumes survival probabilities `a_{r,j} = Π c_i`; a better-selected path has a flatter survival curve, directly improving the scheduler's admission quality. Two DSpark-line distillates compose.
4. **× tree re-gate (Issue 717, armed):** the selector is the **cheap alternative** to the batched tree-verify harness — if chain acceptance reaches the break-even band, the reopen case no longer depends on tree-verify infrastructure. If both land, the tree remains the ceiling probe and the selector the production seam.
5. **× per-NPC decode economics (riir-ai, noted not developed):** tokens-per-verify-cycle at zero training feeds the 20 Hz tick budget story for NPC token generation (the Research 316 §2.3-3 crowd angle).

### 2.4 Suffix-decay-is-local validates the bigram bet

DFlash 2's two locality diagnoses — "a candidate's fit depends mainly on the token just before it" (justifies **pairwise** scoring) and "suffix decay is mostly a local problem" (one-tap-back kernel recovers most of 10 layers) — are independent published validation of Issue 659's modeling choice: a Markov-1 (bigram) coherence term is the right order for the selection role. This is evidence, not a new mechanism; recorded here so the bigram line's next gate can cite it.

### 2.5 Reframing checks (workflow §1 steps 3–4)

- **Latent-space reframing:** token-level speculative decode; forcing an HLA/functor/shard frame weakens it (same call as Research 316 §2.5 — correctly GOAT/Gain-tier inference primitive, not Super-GOAT latent). The one latent-flavored piece is the entropy gate (§2.3-2), which is already the house sigmoid-projection shape.
- **Game-context reframing:** no new NPC behavior class — same decode loop, better tokens/cycle. Q2 of the novelty gate fails on this axis; the value is decode economics under tick budgets.

---

## 3. Verdict

**Gain** — actionable POC composing three shipped substrates into the DFlash 2 mechanism modellessly; filed as Issue 671.

Not Super-GOAT: Q2 fails (no new capability class — optimization of the existing speculative loop), Q3 is an inference-speed moat item rather than a game-product selling point. Not PASS: the composition is unshipped (signal-diff §2.1), the headroom is measured (0.2984 → 0.8940), and the target band (~0.5–0.7) is named by the G3a economics. Not GOAT-yet: GOAT requires the measured gain — the POC is the gate; promote if it clears.

**Published prior art (novelty check, mandatory §4):** parallel block drafting is crowded — DFlash [2602.06036], DSpark [2607.05147], Domino [2605.29707], P-EAGLE/vLLM. Concept-level selection prior art exists: **EAGLE-2 [2406.16858]** dynamic draft-tree pruning by confidence, Dynamic-Width Speculative Beam Decoding (AAAI 2025), "Optimal Draft Token Selection" (ACL 2025), JetSpec (target-aligned candidate-tree scoring, converging from the tree side). No indexed prior art found for the specific mechanism (low-rank bilinear **adjacent-pair** scorer over a per-position top-k lattice from a parallel drafter + greedy single-path walk) — and DFlash 2 has no arXiv/critique yet (1 day old). For us the claim is narrower and safe: the `U` + bigram composition is unshipped here.

**Redirects (explicit justification):**
- **Two-tap dynamic conv → riir-train, genuinely out-of-scope TODAY:** the conv inserts into a *trained* parallel drafter; the stack trains no DFlash-style drafter (drafters are GGUF-loaded models or modelless tables), and Research 407's Weaver redirect already established the same boundary for the trained-adapter alternative. Recipe item if a parallel drafter is ever distilled (Ternary-DFlash): two-tap dynamic depthwise conv before/after each sublayer, +3% params, base kernel + per-16-channel content correction, position 0 reads the last verified token — cites Dynamic Short Convolutions [arXiv:2606.03825] + Canon Layers [OpenReview kxv0M6I7Ud].
- **Suffix-decay measurement pattern** (within-block attention mass by layer/head — the 30%→8% collapse table): a one-off diagnostic worth reusing if we ever audit a trained drafter's block attention; not actionable now.

**MOAT gate (§1.6, katgpt-rs):** in scope — spec decode is the named transformer-stack slot. Public primitive (generic math over marginals + a corpus table; no game/chain/shard semantics). Correct repo.

---

## 4. POC Sketch (delegates to Issue 671)

Arm set on the Bench 694 harness (proxy corpus table + Bonsai GGUF, offline acceptance first, Metal wall-clock second):

1. **Lattice:** per-position top-m candidates from `dflash_predict_parallel` marginals (real context evidence — the `U` source). Pure-unigram `U` is a recorded negative control expected to degenerate toward bigram-greedy.
2. **Score:** `S_t(a,b) = U_t(b) + λ_t · log P_bigram(b|a)`, `λ_t = λ₀ · σ(−κ·margin_t)` or entropy-sigmoid (the `H(h_t)` analog). λ sweep ∈ {0, 0.5, 1, 2}.
3. **Walk:** greedy from last verified token; optional max-product Viterbi variant (O(K·m²)).
4. **Baselines:** argmax-of-marginals chain; greedy bigram chain (0.2984); tree ceiling (0.8785–0.8940); break-even band 0.5–0.7 (G3a economics).
5. **Gate:** offline acceptance ≥ 0.5 at depth 8 → proceed to the G3a wall-clock re-run; feature stays opt-in (`bigram_markov` family) until then.

## 5. References

- DFlash 2 blog: https://inco.ai/blog/dflash2/ (Inco AI, 2026-08-18)
- DFlash: arXiv:2602.06036 (Z Lab, ICML 2026) · DSpark: arXiv:2607.05147 · Domino: arXiv:2605.29707
- EAGLE-2: arXiv:2406.16858 · Dynamic Short Convolutions: arXiv:2606.03825 · Canon Layers: OpenReview kxv0M6I7Ud
- Internal: Research 316 §1.2 (DSpark contract — confirmed by §1.5 above) · Research 407 §1.1 (acceptance ceiling) · Bench 693/694 (headroom + economics) · Issues 659, 670, 671 · Plan 339
