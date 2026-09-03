# Research 528: MemoryWalker — Conditioning-Tree Training Under Context Compression + the Modelless Consistency Audit

> **Source:** "MemoryWalker: Stop Training Agents on Contexts They Never Saw" — Zinco J, Zhu, Huang, Wang, Xie, Ye (Token Foundry, Alibaba), arXiv:2609.00865, 2026-09-01, CC BY-NC-SA 4.0, **no code released** (checked 2026-09-03).
> **Date:** 2026-09-03
> **Status:** Done — verdict recorded; POC filed as Issue 719
> **Related Research:** 523 (H2O norm-age-normalized KV eviction — the deferred consumer this audit gates)
> **Related Plans:** none filed (training recipe is conditional on the dormant L4 RLVR lane — recorded here, §5)
> **Classification:** Public

---

## TL;DR

Production agent harnesses (Claude Code, Qwen-Agent) compress context mid-rollout, so every eviction **branches the effective conditioning history** — the learning object is a tree, not a sequence. The paper formalizes the two wrong linearizations (rightmost path = *time-travel leakage*; DFS replay = *stale-context leakage*), gives two exact gradient corrections (LogitTree, packed 4D mask), and one single-backward relaxation (SDCC) whose residual per-junction KL yields an `O(sqrt(eps_KL))` train-deploy total-variation bound (Pinsker).

**Distilled for katgpt-rs (modelless, inference-time):** the transferable object is not the training loss — it is the **conditioning-consistency audit**: at any point where a serving path conditions on a *semantically compressed* context (window, eviction, summarization, budget packing), run a two-forward pair (compressed-conditioned vs full-context teacher) and measure the per-junction forward-KL. Pinsker's `TV <= sqrt(KL/2)` is **unconditional** (a pure fact about the two next-token distributions — the paper's Prop 10 proof is exactly this, no train-time assumptions), so the same measurement is a *proven behavioral-gap bound* between context regimes at inference. Nothing in the stack computes a KL/TV distance between conditioning regimes today (grepped: confirmed — see §4); our existing compression gates are binary bit-identity (Bench 756 loop-flips 0/512) or reference-forward equivalence at 0.0 diff (Bench 313). The audit is the missing *distributional* gate for every surface where bit-identity is impossible by construction.

---

## 1. Paper Core Findings

- **Conditioning invariant:** every rollout token must be scored (in training) under exactly the live view it was decoded under — `c_train == c_rollout == View_t(H_<t, E_<t)`. Each eviction `(J_k, E_k)` forks the history into a generation-time leg + compressed spine; K junctions → K+1 root-to-leaf branches; **diverging leaves** = tokens whose live prefix ⊋ final-walk prefix.
- **Two pitfalls, opposite signs** (measured on untrained Qwen3-4B): Naive-Compressed (final walk) under-places — `mu_comp in [-4.0, -1.6]` nats, "the model is taught it knows the weather"; Naive-Full (DFS union) over-places — `mu_full in [+0.2, +0.7]`, "the model learns to read discarded context". No reweighting of a single serialization fixes both.
- **logdiff statistic** (Eq. 1): mean |log p(y_t|c_train) − log p(y_t|c_rollout)| under the *same* weights — a diagnostic with a **calibrated zero** (no-compression floor 0.014, pure kernel numerics). Naive-Compressed inflates ~26× the floor on eviction-heavy batches and does NOT self-correct under GRPO training.
- **Exact fixes:** **LogitTree** = K+1 segmented forwards, per-branch loss masks (each token unmasked on exactly one branch — no double-counted trunk gradients), backbone-agnostic, K+1 backwards (5–20× wall-clock at K=5–20). **Packed 4D mask** = causal ⊕ logical masks over the physical union with per-row position-id reassignment; exact under dense softmax + 5 conditions; **violated by** linear attention, sparse/top-k selection, vLLM/SGLang. Theorem 1: they are the same walk.
- **SDCC:** student = ordinary compressed forward (gradients); teacher = **stop-gradient** policy forward on the reconstructed pre-eviction prefix (trunk ⊕ re-inserted evicted span ⊕ post-junction suffix, Eq. 9); forward KL (sleep-phase/wake-sleep justification — reverse-KL's score-function estimator has unbounded variance on the surplus side and is *blind* on the deficit side, which is Pitfall A itself); leaf-gated (no divergence → loss is exactly the task loss; lambda=0 recovers Naive-Compressed). Full beta-VAE treatment: lambda is beta; too-large beta = posterior collapse (Prop 6), hence warm-up + stop-grad/EMA target; convergence to the zero-KL set at O(1/sqrt(T)) → deployment bias O(T^-1/4). Theorem 3: LogitTree and SDCC optimize the same objective up to a lambda-tunable slack.
- **Black-box recovery:** junctions *detected* as prefix divergences between consecutive request payloads (Claude Code / OpenCode) — no white-box eviction records needed for per-turn LogitTree/SDCC.
- **Results** (GRPO, 7 web-search benchmarks): SDCC best overall (EM 46.0 on TC-RAG vs Naive-Comp 36.7); exact walks pin the drift floor; WideSearch @ Qwen3.7-Air/256×H100: LogitTree +42.9% Pass@4 over base in 15 steps.

## 2. Path 0 Inventory (per-component, per-track)

| Paper component | Track | Coverage in stack | Extraction |
|---|---|---|---|
| Conditioning-tree formalization + diverging leaves | modelless-measurable | No instrument | **YES** — branch count/depth/per-junction divergence are O(1) counters from eviction records; the logdiff calibrated-zero discipline mirrors our Bench 649/802 drift-control practice |
| logdiff (calibrated-zero drift statistic) | modelless | Bench 756 flip gates (binary), Bench 313 (0.0-diff equivalence), `numeric_stability::accept()` bands (Issue 775) | **YES** — distributional upgrade: per-junction forward-KL with the Pinsker TV verdict |
| Pinsker bound `TV <= sqrt(eps/2)` | closed-form math | absent | **YES** — unconditional; ships as the audit's verdict function (katgpt-core, beside `numeric_stability`) |
| LogitTree (K+1 segmented forwards + branch loss masks) | training | riir-train RLVR fixer trainer is the only multi-step-trajectory trainer; no eviction harness | Conditional recipe (dormant lane, §5) |
| Packed 4D mask | training + kernel | **Inference-side cousin ships**: riir-gpu Issue 721 `QwenAttentionTreeGatedCubeCL` (ancestor-masked attention over committed prefix ∪ tree rows, speculative tree verify) + katgpt-core `TreePath`/`gdn_tree_verify` | Covered at inference for tree-verify; training-side packing needs a riir-train-gpu custom masked-attention kernel — costlier arm, deferred with LogitTree |
| SDCC (forward-KL, stop-grad teacher on reconstructed prefix) | training (+ the measured KL is modelless) | No forward-KL-to-reconstructed-prefix anywhere (nearest: RMSD self-distillation reverse-KL proxy; DualLeo teacher-student, bit-identity-gated) | Recipe `loss_sdcc` (§5, conditional); the *measured* KL half is the audit above |
| Black-box junction recovery (payload-divergence detection) | modelless technique | nothing | Potential for agent-harness telemetry; no consumer today — recorded only |

**Track verdicts (separately, per the TTPO rule):**
- **Modelless track: Gain.** The audit primitive is real, closes a measured gap (§4), ships opt-in — but has **no live consumer today**: every shipped KV-compression surface (f16 KV, q8kv, PoT scales) is gated at bit-identity, which is *stronger* than any KL bound; the semantic surfaces (Gemma-4 sliding ring, rt_turbo window+sink, TokenBudgetPacker, future H2O) either have no consumer yet (ring ctors "do-not-pass-yet", H2O deferred at 523, delta-KV gated-deferred at Issue 836) or are prompt-assembly-time (packer). → Issue 719 (POC + reopen triggers), no plan, no default promotion. Honest: our *numeric* gates already exceed this paper's standard; the audit adds value only where compression is semantic by construction.
- **Training track: conditional recipe, NOT a filed plan.** The one qualifying pipeline (riir-train RLVR fixer trainer) is dormant by owner call (L4 fixer UNARMED, decline-default — Bench 039/049, `.docs/10_self_evolve/l4_fallback.md` §5). Filing a GPU-hours plan for a declined lane violates the no-dead-plan discipline. Recipe recorded below; reopen with the lane.

## 3. Fusion (what paper × stack produces that neither has alone)

**The conditioning-consistency audit** = Bench 313's reference-forward methodology (changed conditioning ⇒ measured equivalence vs a full-context reference forward) × Bench 756's paired-gate shape (greedy-stream flip counting alongside a distributional statistic) × `numeric_stability::accept()`'s calibrated deviation bands × the paper's KL→Pinsker-TV bound and calibrated-zero control (run the pair with compression *disabled* to pin the numeric floor before interpreting any gap).

Audit shape (per compression site):
1. **Pair:** one forward under the compressed conditioning (student view), one under the full/teacher conditioning (no grad — free at inference).
2. **Calibrate:** repeat with compression off → numeric floor (the paper's 0.014 discipline; our Bench 649 contention lesson applies — quiet box required).
3. **Measure:** per-junction forward-KL over next-token distributions; total `eps_KL`; **verdict = `TV <= sqrt(eps_KL/2)`** (unconditional Pinsker) + the binary greedy-stream flip count retained as the coarse axis.
4. **Tree telemetry:** junction count, diverging-leaf count, per-junction evicted mass — O(1) from eviction/window records.

Consumers (in reopen-trigger order): (a) **Gemma-4 sliding ring** bounded-vs-unbounded gap across `sw` (Issue 752's truncation-differ pin is existence-only — the audit would *quantify* it when the katgpt-rs ctors get their designated Plan 343 T1.6 consumer); (b) **rt_turbo** window+sink decode; (c) **TokenBudgetPacker** — rank pack budgets by measured audit KL instead of byte count (a modelless selector for riir-rag); (d) **H2O eviction** (Research 523) if un-deferred — hit-rate alone is insufficient, this is the gate.

Cross-links: our speculative tree-verify (Issue 721) already ships the *mask mechanism* (per-row visibility over branch-structured histories) at inference — the paper's contribution beyond it is the training-gradient application under *eviction* (not rollout-branching). Issue 731 (Q4_K `get_window` interleaved-vs-concatenated KV) is the incident class this audit's calibration arm would have caught as a distributional signature rather than a bit-exactness accident.

**Game-context reframe:** the two-brain doctrine already runs this paper's insight as design — an NPC think-brain conditions on a compressed/stale summary (zone KG triple + decayed `last_known_pos`) while the info brain holds the raw history; a decision made under a prefix that was later "evicted" is exactly a diverging leaf, and divergence-by-design is our stated emergent-behavior source. The paper adds nothing actionable to the game loop (NPC cognition is not a trained LLM under harness compression) — recorded as conceptual alignment only.

## 4. Consumer Re-Frame (priority ladder question: "what does this do for the healer?")

**Answer: nothing live; one conditional.** The healer's modelless path never trains; retrieval packing (TokenBudgetPacker → 256K budget) compresses *prompt assembly*, and the only trained consumer of packed contexts is the L4 fixer — dormant (decline-default, owner call 2026-08-28). The audit would become a healer concern only if the fixer revives AND its serving context regime diverges from training. Recorded in Issue 719's trigger list; not counted as current adoption.

## 5. Training-Track Recipe (conditional, dormant lane — do not file a plan from this without the trigger)

For `riir-train`'s RLVR fixer trainer (`bonsai_clippy_l4_rlvr_train` / `clippy_l4_rlvr_train` — the one pipeline with (a) multi-rollout GRPO over tool-call-shaped trajectories, (b) front truncation + `max_bwd_seq`, one flag from (c) live harness eviction):

- `RolloutRecord` gains `EvictionEvent { at, compressed_len }` — the harness is ours, records are free (the paper's black-box recovery is a bonus we don't need).
- `loss_sdcc.rs` (beside `loss_asft.rs`, feature `sdcc`): existing advantage-scaled policy grad **+** `lambda * sum_e forward-KL(stop-grad teacher on uncompressed prefix ‖ compressed-conditioned student)` over post-eviction tokens. One extra no-grad teacher forward per eviction; **backward count unchanged** — composes with the single-backward ternary path, zero new kernels. lambda: warm-up 0 → 0.1 over first 20% steps (paper's schedule; do not tune per-harness, their stated anti-overfit rule); **do not exceed the collapse threshold** (Prop 6).
- Degeneration: zero evictions ⇒ bit-identical to current GRPO (G3 by construction). LogitTree (K+1 backwards) and the packed mask (custom riir-train-gpu kernel) are the exactness fallback arms — deliberately second on Bonsai-27B@4090 economics.
- Cost: +40–60% wall on eviction-heavy steps (~12→18 h per 200-step run; ~2× total for the trained-vs-naive A/B).
- GOAT: G1 micro-GPT + synthetic evictor, train-deploy logprob gap ≤ sqrt(eps_KL) bound vs naive's unbounded inflation, deterministic ×3; G2 trained-vs-naive A/B on held-out verifier-pass reward, stratified by evictions/batch (SDCC ≥ naive, margin on the heavy stratum); G3 eviction-free stratum ≤1% delta; G4 alloc-free loss path; G8 standing fixer never-wrong gates.
- **Trigger to file the real plan:** L4 fixer lane revival, or any RLVR run with live context compression in the serving loop.

## 6. Prior-Art Record (§4 searches, 2026-09-03)

- **Prompt Trees (Scaled Cognition blog, 2025-12-01)** — prompt-tree linearization + block-sparse FlexAttention mask giving exact per-path gradients: **mechanism-level prior art for the packed-tree-mask half** (efficiency across shared prefixes, not eviction; no leakage analysis, no LogitTree, no SDCC, no bound). The paper itself does not cite it; our note must not claim mask novelty.
- TITO (Gallouédec & Rasul 2026, HF blog) — token-level consistency baseline the paper extends; Slime stack provenance acknowledged in-paper.
- MemAgent (2507.02259), ReSum/FoldAct, MEM1, AgentFold, MemexRL, Memory-R1 — problem-space neighbors (memory-conditioned agents); none formalize the conditioning invariant or the corrections.
- Training-inference mismatch numerics (arXiv:2602.01826) — engine-numerics axis, orthogonal to conditioning.
- **No prior art found** for eviction-exact gradients, the per-junction stop-grad-teacher KL, or the TV bound as specified. No code released for this paper.

## 7. Panel Record

Four agents spawned in one parallel batch (prior-art web search; codebase grep across 6 repos with vocabulary translation; No-GD advocate; model-based advocate). Merged: No-GD returned the 5-extract audit family (items 1/3/4 adopted here; item 2's "bound is heuristic at inference" caveat is **overturned** — Pinsker is unconditional, the paper's own Prop 10 proof cites no train-time assumption; what changes at inference is only the *interpretation* of the gap, not the bound's validity); model-based advocate audited 4 training surfaces and qualified exactly one (the RLVR fixer trainer, §5); grep agent established the no-KL/TV-instrument gap and the bit-identity-stronger fact for numeric surfaces; prior-art agent returned the Prompt Trees caveat. Brief hygiene held (repos described by shipped file/type names).

## 8. MOAT Gate + Verdict

- **katgpt-rs:** in-scope (KV/serving-gate slot, beside the quant/determinism gates). Audit primitive = fundamental gate *design*, but tier is capped at **Gain** by the novelty-vs-consumer rule: no live consumer, all shipped compression gated stronger (bit-identity), so no GOAT gate can run today and no promotion path exists yet. Opt-in `cond_audit` POC per Issue 719.
- **riir-ai / riir-train / others:** no filing. Game lane: conceptual alignment only (§3). Training lane: conditional recipe (§5), trigger-gated.
- **Honest limits:** the paper's "26× drift" is *their* harness under *their* eviction densities — no analog number is claimable for our surfaces without running Issue 719; the exact-walk methods (LogitTree/4D) are irrelevant to us until a training-consumer trigger fires; black-box junction recovery has zero consumers.

**Files:** `.research/528` (this note) · `.issues/719_conditioning_consistency_audit_poc.md` (POC + triggers) · cross-ref appended to `.research/523`.
