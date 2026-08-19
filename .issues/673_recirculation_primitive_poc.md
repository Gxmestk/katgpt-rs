# Issue 673: Recirculation primitive — cross-step residual mixture operator + Gemma2 PoC

> **Source:** Research 492 (arXiv:2608.17981 "Recirculation" — Mozer et al., Google DeepMind; paired with arXiv:2608.08888 "Full-bandwidth transformer").
> **Filed:** 2026-08-19
> **Priority:** high (immediately-actionable modelless half; the paper shows Gemma2-family gains as pronounced as Gemma3 — we hold gemma-2-2b-it-f16.gguf)
> **Relationship to Plan 431:** direct follow-up. Plan 431's defend-wrong PoC refuted the fixed-pair overwrite (`RelocateOp` CLOBBERS in 2/4 clean configs). Recirculation's convex norm-matched mixture is the semantics that failure mode predicts would fix it. Same stage topology, different mixing semantics, plus the cross-step axis.

## Why

The vertical feedback channel of a decoder-only transformer is narrow: only the sampled token returns to the bottom of the stack; deep-layer conclusions are depth-frozen. Recirculation (training-free) leaks `α·L2norm-matched(source deep-layer state)` into a shallow destination layer at the NEXT input step — 4.7–16% ppl reduction on off-the-shelf Gemma3 (up to 35% at 12B; 23% with the adaptive variant). Not shipped by us: `cross_stage_relocation` is same-pass overwrite; `LoopMode::TrainingFree` is depth-only. See Research 492 §2.2 for the signal-diff.

## Tasks

### Phase 1 — operator (katgpt-core, `recirculation` feature, opt-in)

- [ ] `RecircOp { src_stage, dst_stage, alpha, beta, norm_match: bool, ramp_ticks: u32 }` — cross-step convex-mixture operator on stage outputs; sibling of `RelocateOp`.
- [ ] Mixture math: `z' = α·(‖z_d‖/‖z_s‖)·z_s + β·z_d`, β = 1−α (convex) and β = 1 (non-convex, larger models per paper App. B.3) variants.
- [ ] Ramping schedule `α_t = min(t/ramp, 1)·α` (early-position harm at 1B, paper §4.3).
- [ ] Layer-pair heuristic constants: dest band 0.15–0.33L ← src band 0.42–0.73L (paper Table B.1: {11,4} 1B / {18,9} 4B / {35,16} 12B).
- [ ] Unit tests: mixture boundedness, norm-matching (identity when norms equal), ramp schedule, β=1 variant, zero-alloc steady state (fixed scratch, no per-step heap).
- [ ] GOAT G2: per-step mixture overhead ≤ 1µs at D=2048 (it is O(D) — 2 norm reductions + axpy).

### Phase 2 — defend-wrong PoC (riir-poc; MANDATORY before any promotion, §3.6)

- [ ] Harness: gemma-2-2b forward (riir-ai loader), ~500×1024-token windows from 2+ datasets (arXiv + PG19 style).
- [ ] Arms: (a) baseline no-recirculation; (b) recirculation fixed α ∈ {0.07, 0.10, 0.15} with swept layer pairs; (c) **R417 overwrite on the same pairs** (expected: clobbering — reproduces Plan 431's failure on a real model, the contrast arm); (d) recirculation + temperature 1.2 (additivity control, paper §4.4.2: effects ≈ additive, not temperature artifact).
- [ ] Gate: ppl reduction > 0 on ≥2 datasets for (b); (b) strictly safer than (c) at equal layer pairs. Record raw numbers in Research 492 §"PoC Addendum" whichever way it goes.
- [ ] Honest cost measurement alongside: decode = 2 stack instances/step (serial ~2× FLOPs; 2× KV-cache footprint); prefill goes serial. Compare against KVarN-compressed KV to state the real memory delta (R159 interaction).

### Phase 3 — decode-stack integration decision (riir-ai; only on Phase 2 PASS)

- [ ] If the 2×-KV cost is absorbed (KVarN or acceptable at 2B scale): wire recirculation into the Gemma2 decode loop behind a feature flag; measure end-to-end tok/s + ppl.
- [ ] If not: keep the operator substrate-only in katgpt-core; record the cost verdict in the research note.

### Phase 4 — GOAT gate + promote/demote

- [ ] G1 determinism: fixed α + fixed pairs ⇒ bit-identical repeat runs.
- [ ] G2 overhead gate (Phase 1 bench).
- [ ] G3 no-regression: default build untouched (opt-in flag).
- [ ] G4 alloc-free steady state.
- [ ] Promotion decision: default stays OFF until the PoC quality axis passes on OUR substrate (paper numbers are Gemma3-family claims, not ours).

### Phase 5 — (deferred, on PoC success) belief-recirculation guide

- [ ] riir-ai Super-GOAT guide evaluation: L2→L1 next-tick leak for NPC interpretation hysteresis; α as personality knob; salience-gated recirculation (R148 bridge); `decay_confidence` as the scalar limit. Guide obligations per research skill §1.5 trigger only with the quality axis proven.

## Non-goals

- Full-bandwidth training (→ riir-train Plan 344).
- Blockwise recirculation for parallel prefill (paper's own future work).
- The adaptive MLP variant (modelless replacement = sigmoid salience gate, Phase 5 territory).
