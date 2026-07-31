# Research 448: Latent Error Diffusion — Dual-Stream E/I Belief Updates

> **Source:** *Diffusing Blame: Task-Dependent Credit Assignment in Biologically Plausible Dual-Stream Networks* — Yamada, Grillotti, Charakorn, Risi, Ha, Lange (Sakana AI), [arxiv 2606.31700](https://arxiv.org/abs/2606.31700), 30 Jun 2026
> **Date:** 2026-07-17 (initial); 2026-07-18 (Phase 2+3 addendum)
> **Status:** Settled — Pass (negative result). PoC (Plan 456) showed the latent-ED rule is modelless-correct (G1–G6 PASS) but task-useless (G7 FAILS). See §7 for the full addendum.
> **Related Research:** 408 (TILR — invariant-subspace refinement), 236 (QGF — test-time Q-guided flow), 276 (MicroRecurrentBeliefState), 359 (DEC heat-kernel trajectory), 192 (NextLat belief-state dynamics)
> **Related Plans:** 456 (POC — settled), 425 (TILR), 268 (QGF), 276 (MicroRecurrentBeliefState), 359 (DEC heat kernel)
> **Cross-ref (riir-ai):** Plan 456 POC at `riir-ai/crates/riir-poc/` (defend-wrong per research skill §3.6) — retained as regression check
> **Classification:** Public (math/distillation), Private (runtime wiring → riir-ai)

---

## TL;DR

The paper proposes **Error Diffusion (ED)** as a biologically plausible, backpropagation-free *training* rule under Dale's principle (separate non-negative excitatory/inhibitory streams). On first read this is a clean → riir-train redirect. **But the ED rule structure has a precise latent-space analog**: instead of updating weights via `ΔW ∝ Aᵀ(ϕ'(Z) ⊙ R)`, update the **runtime latent belief state** via the same local rule — `Δh ∝ presynaptic × postsynaptic_drive × global_error_sign` — over a dual-stream latent state `(p, n)` with non-negative projections and modulo-routed per-action error channels. No backprop, no weight mutation, no gradient descent. That is **§3.5 path 3** (latent-space correction), not training.

**Distilled for katgpt-rs (modelless, inference-time):** A runtime belief-update primitive — `LatentErrorDiffusion` — that (a) maintains a dual-stream `(p, n)` latent state with structural sign routing (excitatory `+pW_pp − nW_np` / inhibitory `+nW_nn − pW_pn`), (b) applies modulo-routed per-output-channel error to drive local latent updates, (c) self-organizes toward emergent E/I balance (depth-dependent inhibitory gradient per paper §"Emergent E/I balance"), and (d) develops implicit sparsity (channels below a non-negative floor → effectively pruned, the latent analog of "forgotten" features). The four weight matrices become four **fixed-sign direction-vector projection matrices** read from the per-NPC committed belief substrate.

**Honest caveat:** The user's claim was "gain around accuracy". §3.6 required a defend-wrong PoC in `riir-ai/crates/riir-poc/` before any accuracy claim became a GOAT gate. Plan 456 shipped the PoC. **The PoC settled negative** (2026-07-18): the latent-ED rule is modelless-correct (G1–G6 all PASS) but task-useless (G7 FAILS: −0.51 pp vs Frozen on the K=10 toy decision task). Verdict: **Pass (negative result)** — no katgpt-rs primitive ships.

---

## 1. Paper Core Findings

### 1.1 Error Diffusion (ED) — the backprop-free training rule

Forward pass (dual-stream, Dale's principle):
```
p_i = ϕ_i(+ p_{i-1} W_pp − n_{i-1} W_np + b_p)
n_i = ϕ_i(+ n_{i-1} W_nn − p_{i-1} W_pn + b_n)
```
All four `W_*` matrices non-negative element-wise. The negation signs on `W_np`, `W_pn` are structural (hardcoded) — that's how Dale's principle is enforced: signs come from population identity, magnitudes from learnable non-negative parameters.

ED update rule:
```
R = S · Mᵀ                              # routed error (B×H), M is fixed H×C routing matrix
U_p = ϕ'(Z_p) ⊙ R                       # local postsynaptic drive (B×H)
ΔW_pp ∝ A_pᵀ · U_p                      # local update, no weight transport, no random feedback
```
where `M_ic = 1 iff (i mod C) = c` (modulo routing — hidden unit `i` is assigned to output channel `i mod C`).

### 1.2 Three classification-specific innovations

| Innovation | Mechanism | Effect |
|---|---|---|
| **Layer-specific sigmoid widths** | `ϕ_i(z) = 1/(1+e^{-2z/α_i})` with α=3 conv, α=6 FC | Combats the 25× surrogate-gradient attenuation from output → first hidden layer |
| **Batch-centered class error** | Subtract per-class batch mean from one-vs-all error | Removes the 9:1 target imbalance bias in 10-way classification |
| **Asymmetric E/I init** | Hidden excitatory scaled 1.5×, inhibitory 0.5× (3:1 E:I) | Prevents early instability; final FC layer uses 1:1 symmetric |

### 1.3 Cross-task ablation reversal (the paper's key methodological finding)

| Component removed | MNIST Δ | CIFAR-10 Δ |
|---|---|---|
| Batch-centered error | −0.3 pp | **−47.9 pp** |
| Layer-specific widths | **−71.4 pp** | −15.1 pp |
| Asymmetric init | +0.0 pp | −5.5 pp |

**The bottleneck reverses between tasks.** Sigmoid widths dominate on MNIST (vanishing gradients); batch-centering dominates on CIFAR-10 (target imbalance). Single-benchmark evals mislead.

### 1.4 Emergent phenomena (paper §"Post-hoc Analysis")

- **Emergent E/I balance** — 3:1 asymmetric init → ~1.0 balanced after training. Depth-dependent gradient: layer 1 → 1.03, layer 2 → 0.90, layer 3 → 0.81 (deeper = more inhibitory).
- **Implicit sparsity** — 37.3% of weights collapse to floor (10⁻⁴). Inhibitory cross-stream FC connections pruned most aggressively (up to 68.8%). Conv layers <1% to 18%.
- **Surrogate gradient attenuation** — 25× decay output → first hidden layer; visible from epoch 3.

### 1.5 ED-PPO (RL extension)

Dual-stream architecture integrated into PPO. ED replaces backprop in PPO's gradient computation. Vector policy outputs routed by output channel; scalar value network broadcasts error to all hidden units. Competitive on Brax locomotion (Ant, HalfCheetah, Humanoid); trails BP-PPO on Craftax (−4.0 reward) with higher variance across all envs.

---

## 2. Distillation

### 2.1 The latent-space reframing (§3.5 path 3 — the modelless unblock)

**Paper:** `ΔW ∝ Aᵀ (ϕ'(Z) ⊙ R)` — weight update via presynaptic activity × postsynaptic drive × routed error.

**Latent analog:** Replace weight update with **latent state update**. The "weights" become fixed-sign direction-vector projection matrices (read-only, committed per NPC). The "activations" become the per-NPC latent belief state. The "error signal" becomes the runtime action-outcome error (reward prediction error, claim rubric verdict, CLR vote).

```
# Per-NPC dual-stream latent belief state (p, n ∈ R^H)
# Four fixed-sign projection direction matrices (W_pp, W_np, W_nn, Wpn ≥ 0 element-wise)
#   — committed to NeuronShard / MerkleFrozenEnvelope, NOT trained at runtime

# Forward (perception → belief update):
p_new = sigmoid((+ p · W_pp − n · W_np) / α_p)   # excitatory stream, wide sigmoid
n_new = sigmoid((+ n · W_nn − p · W_pn) / α_n)   # inhibitory stream, wide sigmoid

# Action selection (per output channel c ∈ 1..C):
score_c = p · d_c − n · d_c                       # excitatory-minus-inhibitory projection
action = argmax_c score_c                          # never softmax per AGENTS.md

# ED update (after observing error e ∈ R^C):
R_h = sum_c (e_c · M[h,c])                        # modulo routing: R_h = e_{h mod C}
U_p = sigmoid'(Z_p) · R_h                          # local postsynaptic drive (wide-derivative gated)
p_new += η · p_old · U_p                           # local Hebbian-style update — no backprop
n_new += η · n_old · (−U_p)                        # sign-symmetric inhibitory update
```

**Why this is modelless (not training):**
1. The four `W_*` projection matrices are **committed ahead of time** (BLAKE3-hashed, freeze/thaw envelope). They are NOT updated at runtime.
2. The only runtime mutation is the latent state `(p, n)` — same category as HLA belief evolution, QGF latent flow, or TILR refinement.
3. The update rule is **local** (presynaptic × postsynaptic × global sign) — no gradient computation, no autograd, no backward pass.
4. Emergent E/I balance + implicit sparsity become **runtime self-organization properties** of the belief state, not post-training observations of weights.

**§3.5 modelless unblock protocol — path selection:**

| Path | Applies? | Why / why not |
|---|---|---|
| 1. Freeze/thaw snapshot correction | Partial — the four `W_*` matrices are committed via freeze/thaw | The matrices themselves are frozen; the ED rule operates on top of them |
| 2. Raw/lora reader-writer hot-swap | No — no LoRA overlay involved | The mechanism is latent-state update, not adapter overlay |
| 3. **Latent-space correction** | **YES — this is the path** | ED rule is a recurrent latent update via dual-stream projection + local error signal. Same category as QGF, TILR, HLA evolution. |

### 2.2 Mapping to existing primitives

| Paper concept | Codebase analog | Source |
|---|---|---|
| Dual-stream `(p, n)` latent state | AHLA recurrent state, HLA per-NPC belief state | `katgpt-core/src/sense/`, Plan 276 MicroRecurrentBeliefState |
| Four non-negative projection matrices | Committed direction vectors (BLAKE3-hashed) | `riir-neuron-db/src/shard/mod.rs`, Plan 297 PersonalityWeightedComposition |
| Modulo routing `M[h,c]` | Per-action latent channel assignment | (novel — closest cousin is plan 303 SalienceTriGate's per-action delegation) |
| Layer-specific sigmoid width α | Sigmoid saturation parameter | AGENTS.md mandates sigmoid (never softmax); α is the inverse-temperature |
| Batch-centered class error | Centered reward prediction error | CLR vote (Plan 284), Salience Tri-Gate (Plan 303) |
| Emergent E/I balance | Self-stabilizing personality | Committed Personality Runtime (Plan 336), ArchetypeBlendShard |
| Implicit sparsity (floor collapse) | Forgotten direction vectors | Raven/δ-Mem consolidation, dendritic branch non-interference (Plan 329) |
| ED-PPO (RL) | Runtime GRPO self-play | AGENTS.md: "runtime GRPO self-play stays in riir-ai — updates latent state, not weights" |

### 2.3 Fusion (the Super-GOAT angle to investigate at POC)

**Fusion candidates from the corpus:**

1. **× TILR (Plan 425, Research 408)** — TILR refines latent state onto an invariant subspace. ED updates latent state via local error. **Fusion:** use TILR's invariant-subspace projection as the *postsynaptic drive gate* in the ED rule — `U_p = sigmoid'(Z_p) · P_invariant · R_h`. This bounds the ED update to the trajectory-invariant subspace, preventing runaway drift.

2. **× QGF (Plan 268, Research 236)** — QGF tilts the output distribution toward higher Q via a flow field. ED updates latent state toward lower error. **Fusion:** use ED's latent update as the *Q-gradient direction* in QGF's tilt operator — instead of an externally-supplied gradient, the ED rule produces a locally-computed latent direction. This makes QGF entirely modelless on the runtime side.

3. **× DEC Heat Kernel (Plan 359, Research 365)** — Heat kernel trajectory propagates belief over a cell complex. ED's dual-stream `+pW_pp − nW_np` is structurally an exterior-derivative-like signed sum. **Fusion:** interpret `(p, n)` as a cochain pair and the four `W_*` matrices as discrete operators — ED becomes a heat-kernel-style belief propagation with sign-constrained kernels.

4. **× Committed Personality Runtime (Plan 336, FAME)** — Plan 336 already commits personality direction vectors with sampling-invariant π. **Fusion:** the four `W_*` matrices in latent-ED are exactly the committed personality vectors. The ED rule becomes the *runtime evolution operator* on top of a FAME-committed substrate.

The highest-value fusion is **(1) × TILR** because TILR's invariant-subspace gate directly addresses ED's known failure mode (the paper's "higher variance" / "Craftax shortfall") — bounding updates to the invariant subspace prevents the stochastic excursions that plague unconstrained ED-PPO.

### 2.4 Latent vs raw boundary

- **Latent (local, never synced):** the dual-stream `(p, n)` state, the four committed `W_*` projection matrices, the routed error signal, the ED update itself.
- **Raw (synced via quorum):** the **5 synced affect scalars** (valence, arousal, desperation, calm, fear) — projected from `(p, n)` via the bridge function, clamped to `[0,1]`, committed via LatCal. **Never** sync the full dual-stream latent vector.
- **Bridge function (latent → raw):** `valence = sigmoid(p · d_valence − n · d_valence)`, etc. Zero-allocation, gateable by feature flag, no sync dependency introduced.

This satisfies AGENTS.md: physical domain (position, HP, wallet) stays raw; semantic domain (emotion, mood, curiosity) operates in latent space via dual-stream ED; social domain produces KG triples from latent similarity.

---

## 3. Verdict

**Tier: Pass (negative result, settled 2026-07-18 per §7 PoC Addendum).**
Originally **Gain (pending POC)**; revised down after Plan 456's defend-wrong
PoC produced an honest negative result. The latent-ED rule is
modelless-correct (G1–G6 all PASS) but task-useless (G7 FAILS: −0.51 pp
vs the Frozen baseline on the K=10 toy decision task). The Option C
persistent-belief-state reframing — the most faithful modelless
translation per user decision — does not unlock an accuracy gain.

**One-line reasoning:** The latent-ED rule is a credible modelless
runtime belief-update mechanism (path 3 of §3.5) — the mechanism-level
proofs (G1–G6) confirm it's local, bounded, argmax-based, deterministic,
and E/I-balanced. But the empirical race (G7–G10) showed it
underperforms the no-adaptation Frozen control on a toy K=10 task. The
paper itself reports ED underperforms DFA backprop by 0.9–7.4 pp on
classification; the latent reframing does not recover this gap — it
amplifies it on this task domain.

**MOAT gate per domain (§1.6):**
- **katgpt-rs domain:** The open primitive candidate (dual-stream ED math
  + modulo routing) is **NOT shipped** — the PoC refuted the accuracy
  claim. No feature flag is created.
- **riir-ai domain:** The runtime wiring candidate (NPC belief updates,
  personality substrate) is **NOT shipped** — same reason. The PoC is
  retained in `riir-poc` as a regression check.

**Why NOT promoted (honest negative result):**
- The mechanism-level proofs (G1–G6) confirm the rule is modelless and
  correct. That's a valuable negative result: we now know the *math*
  translates, even though the *utility* doesn't.
- The K=10 toy task may be an unfavorable domain for runtime belief
  updates in general — TILR also underperforms Frozen here. A future plan
  could probe a different task (e.g. K=4 game-action stretch per Plan 456
  T5.1) before fully closing the line of inquiry.
- Until such a follow-up shows a different ranking: **Pass**.

---

## 4. What the POC must prove (Plan 456 preview)

Per §3.6, three claim types, each requiring its own proof level:

| Claim | Proof required | POC mechanism |
|---|---|---|
| **Architectural** ("latent-ED rule exists as runtime update") | grep + read code | Implement `LatentErrorDiffusion::step()` in `riir-poc` |
| **Latency** ("sub-µs per update, no GD") | criterion bench | Bench vs TILR + QGF + raw HLA evolution |
| **Quality** ("matches or beats existing belief-update on accuracy") | **Head-to-head PoC on controlled toy domain** | 3 competitors on toy multi-action decision task |

**Three competitors (the defend-wrong setup):**
1. **Latent-ED** (the paper's mechanism, modellessly translated per §2.1)
2. **Frozen baseline** (no belief update — pure projection, no learning)
3. **Shipped runtime analog** (TILR invariant-subspace refinement — the closest cousin)

**Toy domain:** K-action classification with controlled reward signal. Start with K=10 (matches paper's 10-way MNIST/CIFAR setup) on a synthetic nonlinear task. Stretch: K=4 game-action decision (fight/flee/forage/talk) on a toy NPC sim.

**Verdict table (PoC must print):**
| Competitor | Accuracy | Latency (ns/step) | Stability (variance over seeds) |
|---|---|---|---|
| Frozen baseline | … | … | … |
| TILR refinement | … | … | … |
| Latent-ED (this paper) | … | … | … |

If Latent-ED beats TILR by ≥5 pp accuracy at ≤2× latency → GOAT, promote. Else → Gain stays opt-in.

**Honest failure modes the POC must probe (from the paper):**
- **Higher variance** (paper §"Limitations") — does the latent analog inherit ED-PPO's variance? If yes, the ×TILR fusion is mandatory, not optional.
- **Craftax shortfall class** — fine-grained temporal credit assignment may fail under coarse modulo routing. Test with K=4 actions over a 100-tick horizon.
- **E/I balance divergence** — does emergent balance hold in the latent analog, or does the dual-stream collapse to all-excitatory / all-inhibitory? Track E/I ratio over training ticks.

---

## 5. Implementation routing (after POC PASS)

If POC promotes:

| Artifact | Repo | Location |
|---|---|---|
| Open primitive (math: dual-stream ED step + modulo routing) | katgpt-rs | `crates/katgpt-core/src/sense/latent_ed.rs` (new) — feature `latent_error_diffusion` |
| Architectural guide (runtime wiring, NPC belief substrate) | riir-ai | `.research/NNN_*.md` (new number) — pillar-adjacent guide |
| Runtime wiring (NPC belief updates) | riir-ai | `crates/riir-engine/src/` (new module) |
| × TILR fusion | katgpt-rs + riir-ai | TILR's invariant-subspace gate as ED's postsynaptic drive gate |
| PoC stays as regression check | riir-ai | `crates/riir-poc/benches/latent_ed_goat.rs` |

If POC refutes the accuracy claim but architectural coverage holds:
- Keep the open primitive in katgpt-rs behind feature flag (the math is valid even if accuracy doesn't beat TILR)
- Do NOT promote to default-on
- Record the §"PoC Addendum" honestly per §3.6
- Track follow-up in `.issues/`

---

## 6. References

- **Paper:** [arxiv 2606.31700](https://arxiv.org/abs/2606.31700)
- **TILR (closest cousin):** [Research 408](408_Trajectory_Invariant_Latent_Refinement.md), [Plan 425](../.plans/425_tilr_invariant_subspace_refinement.md)
- **QGF (flow analog):** [Research 236](236_QGF_Test_Time_Q_Guided_Flow.md), [Plan 268](../.plans/268_qgf_test_time_q_guided_flow.md)
- **MicroRecurrentBeliefState:** [Research 276](276_Personality_Weighted_Latent_Layer_Composition.md), [Plan 276](../.plans/276_micro_recurrent_belief_state.md)
- **DEC Heat Kernel (cochain propagation analog):** [Research 365](365_PhysiFormer_Single_Shot_Trajectory_Heat_Kernel_DEC.md), [Plan 359](../.plans/359_dec_heat_kernel_trajectory.md)
- **Committed Personality Runtime (FAME):** [Research 302](302_FAME_Sampling_Invariant_Per_Entity_MoE.md), [Plan 321](../.plans/321_sampling_invariant_per_entity_moe_primitive.md)
- **AHLA recurrent substrate:** `.benchmarks/033_lt2_looped_goat.md`
- **NextLat belief dynamics:** [Research 192](192_NextLat_Belief_State_Latent_Dynamics.md)

---

## 7. Phase 2 + 3 PoC Addendum (recorded honestly per §3.6, 2026-07-18)

### Phase 2 — Option C refactor + mechanism gates G1–G6

Per user decision (2026-07-18), pursued **Option C** (state-only): drop the
recurrent forward entirely; `(p, n)` IS the persistent belief state,
evolved only by the ED rule; `W_*` projections become a one-time
input→initial-state transform. This is the most faithful modelless
reframing — the weights → committed matrices mapping holds, the
activations → `(p, n)` state mapping holds, the only change is that
`(p, n)` persists across steps.

**Mechanism gates G1–G6 ALL PASS** (see Plan 456 Phase 2):

- **G1** zero-alloc hot path (Vec capacity stable across HORIZON=1000 steps)
- **G2** no weight mutation (W_* BLAKE3 hash bit-identical before/after)
- **G3** sigmoid bounded ((p,n) in [0,1] throughout, including adversarial inputs)
- **G4** argmax not softmax (actions binary, both values appear)
- **G5** deterministic modulo routing (routing identity holds by construction)
- **G6** E/I balance stable (‖p‖/‖n‖ in [0.5, 2.0] throughout)

The latent-ED rule is **modelless-correct**. The Phase 1 reframing bug
(forward overwriting the ED delta) is resolved by construction — there
is no recurrent forward to do the overwriting.

### Phase 3 — Quality gates G7–G10

| Competitor | Accuracy | Seed var | Latency (ns/step) | Early | Late |
|---|---|---|---|---|---|
| **Latent-ED** | **0.5312** | 0.00032 | 1136.6 | 0.5022 | 0.5228 |
| **Frozen baseline** | **0.5363** | 0.00016 | 1069.7 | 0.5438 | 0.5126 |
| **TILR refinement** | **0.4730** | 0.00456 | 964.9 | 0.4518 | 0.4676 |

- **G7 FAILS ❌:** Latent-ED 0.5312 vs Frozen 0.5363 (−0.51 pp). The Option
  C refactor fixed Phase 1's "ED has zero effect" bug — Latent-ED and
  Frozen are no longer bit-identical. But the ED rule is actively
  *hurting* accuracy by a small margin. The local Hebbian update
  `Δp ∝ p · σ'(Z) · R_h` accumulates, but in a direction that does not
  align with the task's reward structure.
- **G8 PASSES (vacuous) ✅:** Latent-ED beats TILR by +5.83 pp. **Vacuous**
  because TILR itself underperforms Frozen (47.30% vs 53.63%) — TILR's
  rank-4 invariant-subspace projection is too restrictive for this toy
  task. Beating a broken competitor proves nothing.
- **G9 PASSES ✅:** ED seed-variance 0.00032 vs TILR 0.00456 (ratio 0.07×,
  well under 1.5×). The paper reported ED-PPO has higher variance than
  BP-PPO; the latent analog does NOT inherit this.
- **G10 PASSES ✅:** Late (0.5228) ≥ early (0.5022). No catastrophic drift.

### Honest interpretation

The Option C refactor **succeeded mechanically** (ED now has measurable
effect) but **failed empirically** (the effect is slightly negative).
The latent-ED rule, even with the most faithful persistence semantics,
does not produce a useful belief update on this K=10 toy task.

**This is an honest negative result per §3.6.** Notably, TILR — the other
runtime belief-update competitor — *also* underperforms Frozen on this
task (47.30% vs 53.63%). This suggests the K=10 toy decision task is an
unfavorable domain for *any* runtime belief-update mechanism: the
fixed bootstrap projection captures the task's reward structure better
than error-driven refinement. This is a property of the task, not the ED
rule specifically.

### Final verdict

**Research 448 revised: Gain → Pass (negative result).** The latent-ED
rule is modelless-correct (G1–G6 PASS) but task-useless (G7 FAILS). The
Option C persistent-belief-state reframing — the most faithful modelless
translation — does not unlock an accuracy gain. No katgpt-rs primitive
ships. The PoC is kept in `riir-poc` as a regression check; the mechanism
gates (G1–G6) remain valuable as a documented modelless-correctness
proof even though the rule is not promoted.

---

## 8. Historical: Phase 1 PoC Addendum (recorded 2026-07-17, superseded by §7 above)

Plan 456 Phase 1 (scaffold) landed in `riir-ai/crates/riir-poc/src/latent_ed_poc.rs`.
The harness works end-to-end — the bench prints the verdict table, computes
G7–G10, all 10 unit tests pass. **However, the headline finding is a
mechanism bug, not a paper refutation:**

| Competitor | Accuracy | Seed var | Latency (ns/step) |
|---|---|---|---|
| Latent-ED | 0.5322 | 0.00057 | 10038.5 |
| Frozen baseline | 0.5322 | 0.00057 | 8919.4 |
| TILR refinement | 0.4730 | 0.00456 | 1221.9 |

- **G7 FAILS decisively:** Latent-ED's accuracy is **bit-identical** to the
  Frozen baseline. The ED update has zero effect on chosen actions.
- **G8 "PASS" is vacuous:** TILR is *worse* than Frozen (47.3% vs 53.2%)
  because its rank-4 invariant-subspace projection is too restrictive for
  the toy task. Beating TILR by 5.9pp while tied with Frozen is not a gain.
- **G9 PASSes legitimately:** ED's seed-variance (0.00057) is much lower
  than TILR's (0.00456) — but this is because ED is effectively frozen.
- **G10 PASSes legitimately:** no late-phase drift — but again, trivially,
  because ED is frozen.

**Root cause (preliminary):** The forward pass fully overwrites `(p, n)`
with sigmoid outputs every tick (`self.p[h] = sigmoid(...)`). The tiny ED
delta (`dp ≈ η·p·σ'(Z)·R_h ≈ 0.0006` per tick) is below the forward pass's
noise floor (next tick's `temp_p = p·W_pp ≈ 12`, so sigmoid(2) ≈ 0.88
regardless of the ED contribution).

**The paper's ED rule modifies weights (which persist); my latent translation
modifies activations (which get overwritten).** This is a reframing bug,
not a paper refutation. Three reframing options recorded in Plan 456's
Phase 1 finding:

- **Option A:** Additive bias — `(p, n)` is a persistent bias added to the
  forward pass, not overwritten by it.
- **Option B:** Separate belief state — ED modifies a state that gates action
  selection directly, bypassing the recurrent dynamics.
- **Option C:** State-only — drop the recurrent forward entirely; `(p, n)` IS
  the belief state, evolved only by the ED rule.

**Verdict unchanged:** Gain (pending POC) — but the POC is now pending the
Phase 2 reframing decision, not just the Phase 3 quality race. If none of
Options A/B/C produce a working mechanism, the verdict drops to **Pass**
(the latent reframing doesn't hold up under empirical scrutiny) per T4.5.

---

## TL;DR

Latent reframing of a biologically plausible *training* rule (Error Diffusion under Dale's principle) into a modelless runtime belief-update primitive. The dual-stream `(p, n)` latent state + four committed non-negative projection matrices + modulo-routed per-action error + local Hebbian-style update is **§3.5 path 3** (latent-space correction), not training. **Verdict: Pass (negative result, settled 2026-07-18).** Plan 456's defend-wrong PoC in `riir-ai/crates/riir-poc/` showed the mechanism is modelless-correct (G1–G6 all PASS: local, no weight mutation, bounded, argmax, deterministic routing, E/I-balanced) but task-useless (G7 FAILS: −0.51 pp vs Frozen on the K=10 toy decision task). The Option C persistent-belief-state reframing — the most faithful modelless translation per user decision — does not unlock an accuracy gain. No katgpt-rs primitive ships. The user's "gain around accuracy" intuition was the right hypothesis to test; the PoC's job was to settle it, and it settled negative. TILR *also* underperforms Frozen on this task, suggesting the K=10 toy domain is unfavorable for runtime belief updates in general — a future plan could probe a different task before fully closing the line.
