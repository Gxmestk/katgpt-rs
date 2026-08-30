# Research 519: GRT — Gated Recurrent Transformers (Recurrent Modulation Depth)

> **Source:** [Gated Recurrent Transformers: Expressive Depth through Recurrent Modulation](https://arxiv.org/abs/2608.15062) — Amr Hegazy, Amr Alanwar, Mostafa Elhoushi, arXiv:2608.15062v4, 26 Aug 2026. Code: github.com/Amr-Hegazy1/gated-recurrent-transformer
> **Date:** 2026-08-30
> **Status:** RECORD — both tracks landed this session (modelless: katgpt-rs `.issues/698`; training: `riir-train/.plans/364`)
> **Related Research:** 073 (LT2 — the weight-shared loop we ship), 097 (Training-Free Loop), 273 (ELT Any-Time — the family map + MoR redirect), 414 (Fully-Looped Readout Blind Spot + loop-stability gap table), 343 (System-1.5 depth-step shortcuts), 442 (LOTUS looped supervision PASS)
> **Related Plans:** 108 (LT2 Looped Pipeline — shipped, GOAT 8/8, default-on), 136 (Training-Free Loop Wrapper — shipped), 428 (Loop-Stability PoC — shipped, `LoopStabilityMode`), 304 (GainCost loop halting — shipped), 283 (Self-Advantage Gate); **riir-train Plan 364 (GRT recurrent-depth recipe — filed this session)**
> **Classification:** Public

---

## TL;DR

GRT trains a **fixed prelude → single shared core iterated R times → fixed coda** transformer where each recurrence step reads a GRU-style **elementwise update gate** conditioned on `(LN(h(r-1)), LN(h_pre))` plus **per-step resampled noise**, and blends convexly: `h(r) = g⊙h(r-1) + (1−g)⊙o(r)`. Result: 3 unique layers match 12-layer GPT-2 Small at isoFLOPS; uniform depth sampling during training yields **emergent early exit with zero auxiliary losses**. For us this is the **5th paper in an already-shipped family** (LT2/ELT/MoR/Geiping/RRT) — the architecture class ships (Plan 108). The value is in the **measured deltas**: a load-bearing signal-diff against our own `ResidualGate`, the strongest evidence yet that the loop anchor must be FIXED (not drifting), a static transcribable gate schedule, halter constants (l_min=2, concave gain), and a **simpler-than-ILSD** any-time training trick (uniform depth sampling, no aux loss). **Verdict: Gain on both tracks** — no new capability class; modelless corollaries → `.issues/698`, training recipe → riir-train Plan 364.

---

## 1. Paper Core Findings

### 1.1 Architecture (p + b×R + c notation)

```
prelude:  h(pre) = Blocks_1..npre(h(0))                    # fixed context encoder, computed ONCE
per step r (1..R):
  h̃(r) = Wproj[h(r-1) + εx, h(pre)]                       # concat(d, 2d) → project; εx ~ N(0, 0.1²) RESAMPLED EVERY STEP
  o(r) = B_shared(h̃(r))                                    # SAME weights every step
  g(r) = σ(f_gate([LN(h(r-1)), LN(h(pre))])/τ + εg)         # 2-layer MLP SiLU, hidden d, bias-init +4 → g≈0.98 at init
  h(r) = g⊙h(r-1) + (1−g)⊙o(r)                             # CONVEX blend (state stays in conv{prev, out})
coda:     logits = LM(Blocks_coda(h(R)))
```
Configs: small `1+1×10+1` (3 unique), medium `2+5×4+2` (9 unique), large `1+5×6+5` (11 unique). Training samples `r ~ Uniform{1..R}` per step (depth sampling).

### 1.2 Results

| Regime | Result |
|---|---|
| isoFLOPS small | GRT 3.145±0.004 vs dense-12L 3.188±0.056 — **beats dense with 72% fewer params**; RRT 3.143 (parity) |
| isoFLOPS med/large | dense leads by 0.05–0.06 nats at standard budget; gap closes at 2× tokens; GRT beats MoR/RRT/Ouro/Poisson by 0.06–0.16 nats at med+large |
| isoPARAMS med/large | **GRT 2.76/2.65 vs dense 2.84/2.71** — deeper recurrence wins at matched params+data (+2.10 pts avg on 9 benchmarks) |
| Serving | large: −62% params, −59% peak decode memory, **+10% compiled latency** (+23% eager — half is gate elementwise launch overhead) |
| Early exit | **emergent** (no aux loss): 92% accuracy at half the steps; loss(k) vs equivalent depth dominates dense early-exit at matched FLOPs; extending R past training-R slightly DEGRADES (converged by R) |

Beaten baselines: MoR (token-level recursion router — needs routing machinery), RRT (per-recurrence LoRA deltas — input-INDEPENDENT diversity, fixed at training time), Ouro (per-iteration supervised loss), heavy-tail Poisson (Geiping-style). GRT is the only column with full sharing + input-dependent gating + variable inference depth + per-step noise (paper Table 8).

### 1.3 Ablation ladder (small, 20k steps — the component value ordering)

| Component | Δ nats |
|---|---|
| recurrence alone | **+0.107 WORSE than dense** (undifferentiated sharing collapses) |
| + prelude/coda bracketing | −0.035 |
| + per-step state noise σx | −0.018 |
| + prelude re-injection (anchor at every step) | −0.022 |
| + elementwise gate | **−0.048 (single largest)** → net 3.141, below dense |

**Anchor ablation (Table 11, trained gate kept active):** fixed `h(pre)` 2.68 | drifting `h(r-1)` 3.38 (**+0.70 nats**) | raw input embedding 3.73 | zeros 8.08. *The anchor must be computed once and frozen — a drifting anchor is worth +0.70 nats even with a learned gate.*

### 1.4 Mechanistic findings (the transferable laws)

- **Write-early, copy-late:** effective gate openness ρ(r) = ‖applied update‖/‖proposal‖ declines monotonically 0.182 → 0.066; copy-saturated dims (g>0.95) grow 19.7%→28.8%; write-saturated (<0.05) negligible throughout — the model prefers selective blending over state replacement.
- **Contrastive projection:** Wproj's state-half and anchor-half have mean row cosine **−0.19** — it reads the *difference* (what changed since the prelude), cancelling stable features. Supplies the mechanism explanation for why the fixed anchor helps.
- **Gate attribution shifts** state-ward (59/41 → 74/26 by step 6), structurally primed by Wx leading singular value +44% vs Wh.
- **77% of total loss reduction lands in the first 2 recurrence steps**; per-step gain is concave (peaks step 2); KL to final output ≈ 0.014 by step 5 — output committed by step 4.
- **Implicit difficulty routing:** hardest token decile gains 10× more across steps than easiest (R²=0.998) — no explicit router; the gate conditions on state, so uncertain positions stay write-heavy.
- **Head specializations are frozen by weight** — the same previous-token/broadcast/self heads fire at every step; the gate, not the heads, modulates per-step contribution. CKA: steps 2–6 form one tight cluster peaking at baseline layer ~11 — recurrent depth compresses the path without replicating deep-layer hierarchy.
- **KV averaging across steps beats full per-step cache:** 0.39× memory AND +0.25 HellaSwag over full R× cache (last-step-only is worst, −5.6). Averaging acts as a mild regularizer.
- **Robust hypers:** gate bias surface shallow (−2..+4 within 0.019 nats at full horizon); σx optimum [0.05, 0.1] (removing costs +0.018); τ=1.0 optimal.

---

## 2. Path 0 — two-track decomposition (per-component inventory)

| # | Paper component | Track | Coverage in stack | Signal-diff (mechanism level) | Extraction |
|---|---|---|---|---|---|
| C1 | Input-conditioned elementwise gate `σ(f([LN h, LN h_pre])+ε)` | training | **Partial** — `ResidualGate { gates: Vec<f32> }` (`katgpt-types/src/enums.rs`) is a **static per-loop elementwise parameter** | **Signal consumed: NONE vs state+anchor.** Ours is a learned constant indexed by loop position; GRT's reads the current state and the fixed anchor per token per step, and injects train-time noise. Same storage shape, different signal class — the §3.6 name-cousin trap exactly. | Gate *form* transfers modellessly as a static convex schedule (C1a below); the input-conditioned version requires training → Plan 364 T5 |
| C2 | Fixed-anchor re-injection (h_pre at every step) | modelless | **No** — `forward_looped` applies the loop directly to the input rep; no anchor concept. Nearest: Plan 428's FLT_res (distribute **prev** state to layers — the DRIFTING anchor GRT refutes with +0.70 nats) | Anchor source: drifting prev-state (428) vs frozen prelude output (GRT). GRT's Table 11 is direct evidence for the frozen variant. | ✅ Issue 698 T2 (`LoopStabilityMode::FixedAnchor`) |
| C3 | Per-step resampled noise (σx state, σg gate) | both | **No** | Pure additive runtime perturbation — modelless-injectable with BLAKE3-seeded determinism; paper gain is smallest (+0.018) and measured on noise-trained weights → expect wash | Issue 698 (deferred, afternoon-cost) |
| C4 | Convex blend form `g⊙h + (1−g)⊙o` | modelless | **Partial** — our gate is additive `h̃ + ρ⊙h_prev` (unbounded growth — the exact instability 428 fights); `sub_step_damped_euler` (Plan 136) is the uniform-g degenerate case | Convexity gives a **free norm bound** ‖h(r)‖ ≤ max(‖prev‖, ‖out‖); additive form has none. Form mismatch risk on checkpoints trained additive. | ✅ Issue 698 T3 (schedule interpretant + spec test) |
| C5 | Uniform depth sampling r~U{1..R} → emergent early exit | **training** | **Noted, never filed** — 414 §4.5 item 3 already flagged "stochastic loop count during training" (citing 2606.29983) as a riir-train item; 273 routed ELT's ILSD → riir-train (also never filed) | GRT's version is **strictly simpler than ILSD**: sample r, no teacher/student split, no λ curriculum, no aux loss. ELT's ILSD = 3-term loss; GRT = just uniform sampling. Validates + simplifies both open backstops. | ✅ Plan 364 T1 (NextLat `rollout_steps` is a fixed-depth field begging for this A/B) |
| C6 | loss-vs-depth k=0..R eval | training | **No** — edge_lora ships the width-axis twin ("width-1 ≥ 95% of width-4"); no depth-axis eval exists | Axis twin of a shipped gate form — copy the gate shape verbatim | ✅ Plan 364 T2 |
| C7 | isoFLOPS + isoPARAMS dual-regime protocol | training | **Partial** — xhc phase-4 protocol holds D/L with a flagged 9× param asymmetry, no param-matched control | Control set: minimal-dense only vs minimal-dense + isoPARAM-dense. GRT's own honesty datum (2.77 vs 2.71 = the isoPARAM-matched loss) shows why both are needed | ✅ Plan 364 T3 |
| C8 | Emergent-exit runtime analog: probe-based early exit, l_min=2, concavity floor | modelless | **Partial** — GainCostLoopHalter (Plan 304) ships step-size+oscillation probes; paper adds the floor constants + the concavity law + the "exit is a *quality* rule" datum (beyond-R degrades) | Halter signal: gain/cost scissors (ours) vs fixed-point contraction (paper's convergence metric). Complementary policies over one kernel; fixture picks per model | ✅ Issue 698 T4 (gated on T1's measured spectrum — pin nothing without it) |
| C9 | KV averaged across recurrence steps | modelless | **Partial** — `tf_loop.rs` ships `CacheStrategy::{Last,First}` + a dedicated stash pass; no Mean variant | Stored statistic: last/first vs running mean. Paper: mean beats full R× cache on BOTH axes | ✅ Issue 698 T5 (lossy-surface gate: argmax stability + max_abs band — the Bench-773 lesson: max_rel with floor denominator cannot certify lossy numerics) |
| C10 | Offline gain-spectrum → committed budget ladder | modelless | **No** — `set_active_budget` consumes tiers; no measured per-model Δ(r) table feeds it | The measurement half of the paper's compute-quality dial; fully modelless | ✅ Issue 698 T1 (de-risks T4/T8; one bench) |
| C11 | Per-token difficulty routing (entropy → loop budget) | modelless | Partial (entropy→tier plumbing exists in `latent_functor/action_verify_gate.rs`; PABEE-class prior art [unverified — quota]) | Gated on a separation probe — if our marginal-gain separation ≈1×, dies honestly | Issue 698 deferred |
| C12 | Hand conditional gate `σ(β·(cos(h, h_pre)−θ)+b)` | modelless | **No** | House bridge pattern (dot+sigmoid); mechanism direction (gate opens on divergence) extracted from the paper's contrastive finding. Honest coin-flip on weights that never learned contrastive reading | Issue 698 deferred (behind T1 mechanism gate) |
| C13 | Full GRT pretraining | training | xhc_pretraining suite = the landing surface (dense control arm exists) | — | ✅ Plan 364 T4 (flagship, ≥4M params per Research 418 floor) |
| C14 | Gate-bias init +4 / copy-dominance recipe | training | 3 gated sites (maglev `2σ(G·a_t)` zero-init, xHC write gates random-init, edge_lora sigmoid gates) | Recipe = "bias toward identity, specialize gradually" — each site derives its own identity direction (GRT's +4 means copy-branch; xHC write gates mean gates start CLOSED) | ✅ Plan 364 T5 |

Every row is terminal: filed plan, filed issue, or explicitly deferred-with-reason. No "candidate" rows.

---

## 3. The signal-diff that matters (C1 — §3.6 discipline applied)

Our `ResidualGate` (Plan 108) and GRT's gate share the NAME and the storage shape — the exact configuration where a name-level "already ships" verdict would be wrong. One read of both formulas:

- **Ours:** `h(τ) = h̃(τ) + ρ_τ ⊙ h(τ-1)` — ρ is a per-(loop, dim) **constant** learned during training; consumes zero runtime signal; additive form permits unbounded residual growth (the instability Plan 428 exists for).
- **GRT:** `h(r) = g ⊙ h(r-1) + (1−g) ⊙ o(r)` — g is computed per token per step from `(LN(state), LN(frozen anchor))` + noise; consumes two live signals; convex form is norm-bounded by construction; +4 bias init makes it start identity-preserving.

Delta: **dynamic-per-token + context-grounded + bounded vs static + signal-free + unbounded.** The paper's own ablation says the static-recurrence world (their "recurrence only" row, +0.107 nats WORSE than dense) is what undifferentiated sharing buys — which is the cautionary version of our current gate. This does not demote the shipped gate (ours sits on different consumers: inference-time looping of already-trained weights where a learned dynamic gate cannot exist) — it defines the upgrade path for any model WE train (Plan 364) and the bounded-form corollary for the runtime (Issue 698 T3).

---

## 4. Fusion

**GRT × LT2 × Plan 428 × halters = "anchored gated loop"**: inter-loop normalization (428, prerequisite — GRT's gate consumes LN inputs) → fixed-anchor injection (C2) → convex copy-late schedule (C1a/C4) → gain-spectrum-pinned halter floors (C8/C10) → KV-mean (C9). Every rung independently gated; composition ablated in that order. This is a coordination + constants layer on shipped substrate — the honest tier is Gain, exactly as 273 concluded for ELT.

**Training-side fusion:** GRT recipe × xhc_pretraining (dense control arm + phase-test discipline) × NextLat drafter (the fixed-depth field) × maglev gate inits — Plan 364.

**Game-side (step-4 reframe, noted not landed):** the difficulty-routing law + concave per-step gain restate the EVPI/curo-sity economics at loop granularity — spend recurrence only where the belief state is wrong. Kept as a design note (C11 deferral) until the LM-fixture separation probe earns it; LM-token evidence ≠ NPC-cognition evidence.

---

## 5. Novelty gate (scored per track — TTPO lesson: one verdict per track)

**Modelless track:**
- Q1 no prior art? **NO** — family saturated (LT2/ELT/MoR/Geiping/RRT/Loopformer/Encode-Think-Decode; paper's own Table 8). Runtime analogs shipped (Plans 108/136/428/304).
- Q2 new behavior class? **NO** — any-time looped inference ships (Plan 136 + Issue 035 elastic override).
- Q3 selling point? Weak — constants + wiring on shipped capability.
- Q4 force multiplier? Moderate (connects 428 → halters → budget ladder).
→ **Gain.**

**Training track:**
- Q1? **NO** — GRT is itself the 5th+ arch in its family; we claim no arch novelty.
- Q2? **NO** — recurrent-depth training is an established class; we claim recipe applicability.
- Q3? Moderate — "our looped/latent models train with any-time depth for free" is a real capability upgrade for NextLat/maglev artifacts.
- Q4? Yes (NextLat + xhc + maglev + eval infra).
→ **Gain** (below GOAT until Plan 364's gates measure the gain; if depth-sampled NextLat passes its ≥10% depth-k gate with depth-1 parity, that item re-scores).

**§4 prior-art search disclosure:** web search quota exhausted this session (webReader + search MCP both 429; fetch timeout). Searches ATTEMPTED per the hard gate; the Q1 failures above rest on (a) the paper's own six-way competitive table and (b) our shipped corpus — no external-search dependence. Unverified citations carried from the advocate pass are marked [unverified]: PABEE/"BERT Loses Patience" (training-free early exit precedent for C11), YOCO/cross-layer KV reuse (adjacent to C9, not the same op), DEQ (fixed-point theory behind C8). Re-verify on quota reset before any of those three become load-bearing.

---

## 6. MOAT gate

- **katgpt-rs:** in scope (looped-transformer runtime primitives are a shipped pillar — Plan 108). Gain-tier: constants, orderings, and one bounded form upgrade to existing substrate. Promote/demote: nothing promotes; Issue 698 items gate individually.
- **riir-train:** in scope (active moat — training recipes). Plan 364 is the Path-0.5 landing: recipe + GPU-hours + GOAT gate vs modelless/fixed-depth baseline, per the mandate.
- **riir-ai:** no guide — the game-side transfer is a two-line design note (C11), below guide threshold until measured.
- **riir-chain / riir-neuron-db:** no landing. (C14 pattern-residency note from the advocate — looped-ε-fixed-point commitment — has no live consumer and latents don't cross the commitment boundary; recorded here so it isn't re-derived.)

---

## 7. Verdict

| Tier | Modelless track | Training track |
|---|---|---|
| Super-GOAT | ✗ | ✗ |
| GOAT | ✗ | ✗ (until Plan 364 gates pass) |
| **Gain** | **✓ — note + Issue 698** | **✓ — Plan 364** |
| Pass | — | — |

**One-line reasoning:** the 5th paper in a family we already ship (LT2, any-time loops, loop-stability PoC, gain/cost halting) — no new capability class on either track; the value is a decisive signal-diff on our own gate (static/unbounded vs input-conditioned/bounded), the strongest evidence yet that loop anchors must be frozen not drifting (+0.70 nats), a transcribable copy-late schedule with a free convexity bound, halter constants (l_min=2, concavity, exit-as-quality-rule), KV-mean across steps, and a strictly-simpler-than-ILSD any-time training trick landing on a NextLat config field that exists today.

**Files this session:** `katgpt-rs/.research/519` (this note) · `katgpt-rs/.issues/698_grt_anchored_gated_loop_upgrades.md` · `riir-train/.plans/364_grt_recurrent_depth_recipe.md` · cross-ref lines added to 273 + 414.

### Defend-wrong status (§3.6)

No parity claim is made anywhere: the conditional-gate candidate (C12) is explicitly labeled an honest coin-flip on untrained weights and gated behind T1's mechanism correlation test; the convex-form candidate (C4) carries the form-mismatch risk on additive-trained checkpoints; the halter constants (C8) are gated on T1's measured spectrum with an explicit "if our model puts 40% of gain in step 2, the paper's constant does not transfer — that is a finding, not a failure." Architectural claims (gate signal-diff, anchor ordering evidence) are grep/read-backed. Latent/raw rules: all candidates are latent-space loop ops; nothing crosses the sync boundary.
