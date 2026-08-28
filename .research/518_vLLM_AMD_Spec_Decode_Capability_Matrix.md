# Research 518: vLLM Speculative Decoding on AMD GPUs — Capability Matrix PASS

> **Source:** vLLM blog, 2026-08-23, "Exploring Speculative Decoding in vLLM on AMD GPUs" — https://vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus (first-party measurement study on MI300X gfx942 + MI355X gfx950, ROCm 7.2.53211, vLLM 0.23.1rc1; references EAGLE-3 [arXiv:2503.01840 "Scaling up Inference Acceleration of Large Language Models via Training-Time Test"], DFlash [arXiv:2602.06036 "DFlash: Block Diffusion for Flash Speculative Decoding"], DSpark [arXiv:2607.05147 "DSpark"])
> **Date:** 2026-08-28
> **Status:** Done — PASS (opponent-intelligence record for the perf-rematch league)
> **Related Research:** 002 (Leviathan spec decode), 026 (Gemma 4 MTP), 316 (DSpark — semi-AR + prefix scheduler), 410 (vLLM Dynamic SD — prior vLLM distill), 490 (DFlash 2 — pair-scored selection), 407 (acceptance ceiling), 243 (Bebop acceptance forecast)
> **Classification:** Public

---

## TL;DR

**PASS.** vLLM's capability matrix for five drafting methods (native MTP, Gemma 4 MTP, EAGLE-3, DFlash, DSpark) with matched draft checkpoints from six publishers. Every method is already distilled in our corpus (map in §2); the blog adds no new math. Its value is (a) **opponent intelligence** — the league's fairness manifest must now assume vLLM-class opponents run trained speculators (peak 2.87× DFlash on gemma-4-26B MATH500, 2.68× on Kimi-K2.5), and (b) **four external confirmations** of findings we measured ourselves (§3), plus **one nuance that cuts against a clean story** (§4): drafter-family ranking is checkpoint-dependent, not architectural.

---

## 1. The capability matrix

- Methods served: `mtp` (native), `mtp` (Gemma 4 assistant w/ shared target KV cache), `eagle3`, `dflash`, `dspark` via `--speculative-config`.
- Draft-checkpoint publishers: Google (Gemma 4 assistants), LightSeek (EAGLE-3 for Kimi-K2.5/6/7-Coder incl. MLA variants), Red Hat AI (EAGLE-3/DFlash/DSpark across Llama/Qwen/Gemma/GPT-OSS/GLM/Nemotron/Mistral), Z-Lab (DFlash for Qwen3/3.5/3.6/Gemma 4/Kimi/MiniMax/GPT-OSS/Llama), DeepSeek DeepSpec (all three for Qwen3-4B/8B/14B + Gemma 4 12B), Inferact (MiniMax-M3-EAGLE3, Kimi-K3-DSpark).
- Headline peaks: DFlash 2.87× (gemma-4-26B-A4B MATH500 N=7), Gemma 4 MTP 2.83× (same target MATH500 N=5), DFlash 2.68× (Kimi-K2.5 MATH500 N=7, 310→832 tok/s), native MTP 2.20× (Qwen3.5-122B-A10B MATH500 N=7).
- Tuning instrument: **per-position acceptance heatmaps** (rows = N, columns = draft position) + MAL + AR + end-to-end throughput — the same instrument family as our Issue 742 gates (acceptance 15.15/16 at K=16) and Bebop's `AcceptanceForecast`.
- League note: vLLM's *own* future-work list admits n-gram and suffix decoding are **unbenchmarked** in their matrix — the one drafter family where our online `NgramDrafter` (riir-ai Issue 659; beats static bigram in-domain at K=2, acceptance 0.400 vs 0.087) and `fill_lookup_draft` (Issue 742, 15.15/16) already operate.

## 2. Method → cousin map (all five pre-distilled)

| Blog method | Our record | Status |
|---|---|---|
| Native MTP | 026 (Gemma 4 MTP), 059 (MoE spec-decode co-design), 078 (MTP cluster top-K) | distilled |
| Gemma 4 MTP | 026 | distilled |
| EAGLE-3 | no dedicated note — mechanism (low/mid/high layer concat+projection fused with token embedding, autoregressive draft head) is standard; the actionable layer-capture substrate ships in riir-ai (Issue 749 `prefill_with_all_positions_capture`, 25×; cudarc `forward_token_with_layer_capture`) | summary-only, no math missed |
| DFlash | 490 (+ its DFlash 2 successor) + 000/001 (block-parallel marginal drafting, DDTree) | distilled incl. successor |
| DSpark | 316 — incl. the exact semi-AR contract (parallel backbone base logits + low-rank Markov bias `B = W1[x_{k-1}]·W2`) and the shipped `HardwareAwarePrefixScheduler` (Plan 339, GOAT G1–G5) | distilled + shipped |

## 3. External confirmations (four of our own measured findings)

1. **Suffix decay + non-monotonic N.** Per-position tables: p1 acceptance stays 84–95% flat across N while tails decay to <10% by p10–15; DFlash throughput peaks at N=7 (2.87×) and *regresses* at N=15 (2.40×) on the same target. Confirms Issue 742's knee-at-16 (lookup acceptance 15.15/16; +37%/+32% going K=8→16, nothing beyond) and Research 490's suffix-decay framing.
2. **MAL/AR vs throughput decoupling.** "A method may show higher throughput relative to baseline even with a lower acceptance rate when draft generation is inexpensive." Confirms the DFlash 2 selection finding (490: +0.6% cycle latency buys +21% acceptance length → net win) and Bebop's economics premise.
3. **DSpark semi-AR thesis** (partially — see §4): on Qwen3-8B DSpark dominates DFlash on every workload (GSM8K N=7: 1.63× AR 78.4% MAL 6.49 vs 1.25× AR 54.9% MAL 4.84). Confirms 316 §1.2's "2-layer DSpark beats 5-layer DFlash" on a second target family, and re-confirms the **Issue 717 G2-full contract diagnosis**: the blog states the Markov head "produce[s] a small bias [that] adjusts the base logits produced by the parallel backbone" — reading the trained head standalone (as G2-full first did) is structurally the wrong contract.
4. **Drafter cost can dominate at serving scale.** EAGLE-3 on Qwen3-8B never exceeds baseline on MATH500 (0.44× at N=1 → 0.88× at N=7; baseline 3,530 tok/s aggregate): when the serving baseline is fast, fixed draft+verify overhead eats the gain. Our batch-1 analog is Bench 694/695 (chain-seam 0.326×–0.328× for weak drafters) — different mechanism (sequential verify economics vs stolen batched throughput), same conclusion: acceptance-per-verify-cost is the only currency.

## 4. The nuance that cuts against a clean story

**Drafter-family ranking is checkpoint-dependent, not architectural.** On gemma-4-31B the ordering FLIPS: DFlash beats DSpark on all four workloads (MATH500 N=7: 2.34× AR 68.0% vs 2.20× AR 61.4%; GSM8K 1.95× vs 1.82×; HumanEval 2.02× vs 1.98×; MBPP 1.92× vs 1.84×). So neither "semi-AR > parallel" nor "parallel > semi-AR" holds in general — it is the trained checkpoint's quality on the target that decides. This nuances 316's DeepSeek-checkpoint-based claim, aligns with Research 490 (DFlash 2's pair scorer beats DSpark's Markov correction with 40× fewer params — architecture is a budget, data is the variable), and matches our own Issue 742 result that composing a trained head with our lookup drafter LOSES on doc-repro. Practical rule for any future drafter work: benchmark families per-target/per-workload; never port a ranking across checkpoints.

**Chat absence.** The matrix has no chat acceptance data (GSM8K/MATH500/HumanEval/MBPP; MT-Bench appears once as an N-sweep point). Our Issue 742 chat loss (~41.8 vs vLLM's 123.5–137.4 tok/s, "no chat-capable drafter") is therefore untouched by these tables — and vLLM's own blog does not publish the chat evidence either.

## 5. Training-workflow section — disposition

The blog's speculator-training summary (representative prompts → target-model responses → hidden-state mode **online/offline/hybrid** → train → measure acceptance/throughput, never loss) is an **ops pattern, not new math**: the technical content (DSpark losses α_ce=0.1/α_tv=0.9/α_conf=1.0, position weighting `w_k = exp(−(k−1)/γ)`, STS calibration) already lives in Research 316 §1.2–1.3. One genuinely portable hygiene rule: **responses must come from the exact target model** — "applying the target model's tokenizer or chat template to existing responses does not make the data target-specific."

The one lane this workflow would serve if the owner ever pursues it: a **qwen38 chat-capable drafter** (Issue 742's recorded chat loss; Issue 717 reopen trigger (b) "real-context/trained-head drafter" — the trained DSpark head already banked +17.4% true-acceptance head start there). Substrates exist: hidden-state capture (`forward_token_with_layer_capture` — diagnostic-only at 5 sync + 20 KB dtoh/token, so OFFLINE mode only at current cost; `prefill_with_all_positions_capture` for prefill taps), the `speculative_decode` verify seam (checkpoint/rollback bit-exact), and riir-train GPU capacity. **NOT filed as a plan** — owner-gated new lane; §4's checkpoint-dependence finding says the first step would be a family bake-off (EAGLE-3 vs DFlash vs DSpark heads on OUR workloads), not committing to one architecture.

## 6. Verdict

**PASS.** No new primitive, no plan, no files beyond this record + the PASS-Redirects appended to Research 316/490/410. All five methods pre-distilled; the blog's four confirmations and one checkpoint-dependence nuance are recorded here for grep-ability (search `speculative-decoding-amd-gpus` or the title hits this note).

**MOAT gate (katgpt-rs domain):** neutral-to-positive. The league-intelligence half (vLLM serves 5 drafter families with published matched checkpoints) sharpens the perf-rematch fairness manifest; nothing here reroutes.

---

## Cross-references

- Research 316 (DSpark) — semi-AR architecture, prefix scheduler (shipped Plan 339); this blog's DSpark tables are its first external multi-target replication.
- Research 490 (DFlash 2) — pair-scored selection; §4's checkpoint-dependence strengthens its "architecture is a budget" framing.
- Research 410 (vLLM Dynamic SD) — prior vLLM serving-behavior distill; this blog is the capability-matrix sequel.
- Research 026 (Gemma 4 MTP) — the native-MTP/assistant family in the matrix.
- riir-ai Issue 742 (Bench 754 closeout) — the K=16 knee + chat loss this blog's tables bracket.
- riir-ai Issue 717 (Benches 693/694/695/696) — the G2-full DSpark contract diagnosis, externally confirmed by §3.3.

## TL;DR

vLLM AMD spec-decode matrix: all five drafting methods already distilled (316/490/026/410); PASS. Kept for league intelligence (vLLM + 6 publishers ship trained speculators, peaks 2.87×) + four external confirmations (suffix decay/N-optimum, MAL-throughput decoupling, DSpark contract, draft-cost dominance) + one nuance: drafter ranking is checkpoint-dependent, not architectural.
