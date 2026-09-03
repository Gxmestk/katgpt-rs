# Issue 717 — LT2 deep-loop stability: measure `forward_looped` at T≫4 + runtime damping/tangential stabilization + f32-state contract

**Status:** OPEN — filed 2026-09-03, poc/proof task (no consumer claim yet; `lt2_looped` ships but its deep-loop behavior is unmeasured)

> **Source:** sotaku distillation — `riir-train/.research/440_Sotaku_Late_State_Looped_Solver.md` (pinned `chenglou/sotaku @ 6cdb9a9b`, MIT). **Consumers:** Plan 108 `forward_looped()` (`lt2_looped`, currently T=4), Plan 304 GainCostLoopHalter (gate design), Plan 136/Research 097 (Training-Free Loop — the trained-twin boundary this issue must respect).

## Why

Three measured facts from sotaku (external, 1024-iteration looped inference) have **no in-stack counterpart**, and all three apply to any future deep use of our shipped looped-inference path:

1. **Delayed damping is a runtime, checkpoint-agnostic rescue.** First B iterations untouched, then `h ← (1−α)h + α·F(h)`: rescued a collapsed checkpoint 5.64% → 95.66% @1024 iters; α checkpoint-dependent (0.25 down to 0.03125); costs 1.5–2.3 pts on already-stable checkpoints. Closed-form justification: damping maps a locally-linear mode λ → 1−α+αλ, contracting the unstable radius for α ∈ (0, 2(λ−1)/λ). **Rule from upstream: don't damp unless inference already degrades; sweep α on held-out inputs; check two depths.** Constant-from-iteration-1 damping merely delays collapse.
2. **Tangential-biased correction beats radial.** Scaling the state's radial (magnitude) component alone often WORSENS collapse; scaling the tangential (rotational) update component ×0.25 rescued 3 collapsed checkpoints to 94.7–96.2%. Attribution: the failure is accumulated *direction* drift across answer boundaries, not state size (direction-only readout matches full readout within 0.3pp).
3. **Precision law for carried recurrence state.** Sub-f32 state arithmetic AMPLIFIES with loop depth (their BF16 @4096 = 43.7% vs FP32 98.6%; eager rounding between ops is the killer — compiled chunks retaining f32 intermediates recovered to 92.5%). **Contrast (worth pinning):** riir-ai Bench 802 found f16-KV deviation DILUTES with attention context — attention rounding averages out; weight-tied recurrence accumulates. Any deep-loop serving/training on our stack must keep the carried state f32.
   Plus the **relative-residual trap:** on non-fixed-point recurrences (state RMS grew 24→706 while `‖F(h)−h‖` plateaued ≈0.63), tiny relative residual is a growing-denominator artifact — never use it as a convergence/halting signal for this model class (relevant to GainCostLoopHalter's halt criteria).

## Scope of our gap

`forward_looped()` (Plan 108, GOAT 11/11) is validated at **T=4 only**. Nothing measures T ≫ 4: stability, accuracy-vs-T, state-norm growth, or what a damping/tangential knob does. The looped-model literature upstream of us (Research 073 PASS-redirects) contains no runtime α-swept damping either — FPRM (2606.18206) is architectural (pre-norm), Research 097/SMELT 1/K·1/r scalings are TRAINED constructions. **Boundary:** those stay the answer for trained looped models; this issue is about the frozen-checkpoint runtime path.

## Tasks

- [ ] T1: Instrument `forward_looped` for deep runs — state-norm and logit-finite tripwires per K iterations, opt-in counters (zero cost when off), a `LoopDeepRun` test harness driving T ∈ {16, 64, 256, 1024} on a fixed small model + input suite.
- [ ] T2: Measure the baseline: accuracy/consistency-vs-T and state-norm-vs-T at default settings. Verdict either way is a result (if T=4-installed models are flat to 1024, the guard work is cheap insurance; if they degrade, the knob is load-bearing).
- [ ] T3: Runtime damping knob — `h ← (1−α)h + α·F(h)` after burn-in B, `set_looped_damping(α, B)` or per-call params, feature-gated, default OFF (bit-identical when unset). GOAT: G1 bit-identity at α=0/B=0; G2 on a deterministically-destabilized arm (scaled weights), damping restores sane outputs and α-sweep monotonicity matches the eigenvalue map; G3 stable checkpoints lose ≤ the upstream-measured class only when damping is explicitly enabled; G4 alloc-free in the hot loop.
- [ ] T4: Tangential/radial decomposition probe — decompose Δh = F(h)−h into h-aligned and orthogonal parts; expose scale knobs + a direction-drift diagnostic (direction-only vs full readout gap). Wire into the same gate harness.
- [ ] T5: f32-state contract — pin (test or debug_assert) that the carried hidden state is not rounded through sub-f32 storage across loop iterations in any `forward_looped` arm; document the Bench-802 contrast (attention dilutes, recurrence amplifies) at the site.
- [ ] T6: Halt-criteria note for GainCostLoopHalter: record the relative-residual trap in the halter's docs (never halt on "residual small relative to ‖h‖" for non-fixed-point recurrences).
- [ ] T7: Optional cross-check with riir-train Plan 373's trained artifact (delayed-damping sweep from a second model) — only when that checkpoint exists.

## References

- riir-train Research 440 (§3 damping/tangential/DEQ rows, §4 numerics verbatims) + Plan 373 (training side)
- katgpt-rs Research 073 (looped-LM canonical note — PASS-Redirect appended), Research 035 (Attractor/IFT line — the fixed-point assumption sotaku's DEQ analysis refutes for growing-state recurrences), Research 097/Plan 136 (Training-Free Loop), Plan 304 (GainCostLoopHalter)
- Upstream: sotaku looping/EXPERIMENTS_LOOPING.md + iters/EXPERIMENTS_ITERS.md @ `6cdb9a9b`

## Summary

**(1) Original task:** file the modelless-track residue of the sotaku distillation.
**(2) Accomplished:** issue filed with three extractions (damping, tangential scaling, f32-state contract + residual trap), scope pinned against trained-construction prior art, T1–T7 tasks.
**(3) What remains:** T1–T6 open; T7 contingent on riir-train Plan 373.
**(4) Active plan state:** this issue (OPEN); riir-train Plan 373 (Phase 0 open); riir-train Research 440 (RECORD).
