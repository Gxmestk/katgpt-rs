# Issue 575: PoC — Feedback-Payoff-Weighted Set Attention to Close G8 Collective Inference

**Date:** 2026-08-06
**Source research:** [katgpt-rs/.research/469_collective_intelligence_payoff_schemes.md](../.research/469_collective_intelligence_payoff_schemes.md)
**Source paper:** Wang, Su, Wang, Plotkin (2025) *Individual incentives that promote collective intelligence* PNAS 122(51) e2516535122 — the "feedback payoff" $\pi_A = Y_A(Y - \hat{Y})$ proven Lyapunov-convergent.
**Target primitive:** `set_attention` (katgpt-core, DEFAULT-ON, Plan 354)
**Documented limitation:** [Bench 354](../.benchmarks/354_set_attention_goat.md) L71 — "G8 collective inference FAILED (Super-GOAT→GOAT) — averaging cannot amplify detection; that's a use-case limitation, NOT a primitive defect."
**Status:** Open — PoC task (not a plan, per AGENTS.md "Create issue at .issues for poc, proof, optimization or refactor task, do not create plan").

---

## TL;DR

Set Attention (Plan 354, DEFAULT-ON) explicitly documented a G8 collective-inference failure: plain averaging cannot amplify a detection signal that no individual entity carries strongly enough. The Wang/Plotkin (2025) "feedback payoff" $\pi_A = Y_A(Y - \hat{Y})$ is precisely the credit-assignment shape that converts plain averaging into amplification — proven Lyapunov-convergent and robust to environmental shocks. CLR (Plan 284, DEFAULT-ON) already implements the same math shape `(mean_m v_k,m)^M` and beats majority by +78pp on a 5-cluster fixture.

**This PoC asks:** does feedback-payoff-weighted Set Attention close G8? If yes → promote to a Plan (open primitive in katgpt-rs/.plans/, runtime guide in riir-ai/.research/). If no → honest negative result, leave Set Attention G8 as a documented use-case limit.

Per Research 469 §1.55.2 (BTM-lesson pattern), this is the actionable improvement that makes the paper Gain rather than Pass.

## PoC design

Three competitors in `riir-ai/crates/riir-poc/` (the defend-wrong R&D crate). Use `CARGO_TARGET_DIR=/tmp/issue_575` per AGENTS.md rule and clean up when done.

### Toy domain: synthetic crowd threat detection

- N=64 entities, each holding an 8-dim belief state h_i.
- One entity (the "threat") carries a weak signal: its h_i has a small component along a threat direction d_threat, magnitude ε such that **no individual cosine(h_i, d_threat) exceeds 0.6** — i.e., no single entity can detect the threat at threshold 0.6.
- The remaining 63 entities carry i.i.d. noise plus a small correlated component that, in aggregate, would reveal the threat — but only if the crowd aggregate amplifies rather than averages.
- Ground truth: the threat is entity i*; detection succeeds if max_j crowd_amplified_score(j) > τ for some threshold τ > 0.6.

### Three competitors

1. **Baseline (plain Set Attention)** — `set_sigmoid_attention_into` as shipped (Plan 354). Per G8 documentation: detection F1 ≤ best individual F1 (averaging cannot amplify).
2. **CLR-weighted Set Attention** — each peer j's contribution scaled by `r_j = (mean_m v_j,m)^M` where v_j,m are M=5 binary verdicts from dot-product + sigmoid projection onto M=5 direction vectors (designer-supplied or BLAKE3-deterministic). The `(·)^M` exponent is CLR's headline trick (Plan 284, DEFAULT-ON).
3. **Paper-form feedback payoff** — each peer j's contribution scaled by `w_j ∝ h_j · (d_truth − ĥ)` where d_truth is an external truth probe (designer direction, expectation, or anti-cheat baseline) and ĥ is the current crowd aggregate. This is the paper's exact $\pi_A = Y_A(Y - \hat{Y})$ formula.

### Gate (PoC verdict)

- **PASS (closes G8):** either competitor (2) or (3) achieves crowd detection F1 > best individual F1 by ≥5pp on the toy domain. Promotes to a Plan.
- **FAIL (G8 stays documented limit):** neither competitor beats baseline by ≥5pp. Honest negative result; leave G8 as is. Record raw numbers in Research 469 §"PoC Addendum" per §3.6 protocol.
- **Mixed (latency wins, quality fails):** if quality gate FAILs but latency is sub-25µs at N=64 (matches Set Attention G3), record as "architectural coverage confirmed, quality parity refuted" and leave the follow-up tracked.

## Tasks

- [ ] T1 — Implement the toy domain generator in `riir-ai/crates/riir-poc/benches/set_attention_feedback_payoff_poc.rs` (synthetic N=64 belief states + weak-threat ground truth).
- [ ] T2 — Implement competitor (1): plain Set Attention baseline (just call the shipped `set_sigmoid_attention_into`).
- [ ] T3 — Implement competitor (2): CLR-weighted Set Attention (`r_j = (mean_m v_j,m)^M` scaling on the attention weights).
- [ ] T4 — Implement competitor (3): paper-form feedback payoff (`w_j ∝ h_j · (d_truth − ĥ)` scaling).
- [ ] T5 — Run all three on the toy domain, 1000 trials × 5 seeds, print verdict table (F1, latency, alloc count).
- [ ] T6 — Record raw numbers in Research 469 §"PoC Addendum" (per §3.6 honest-revision protocol — do NOT silently revise the verdict to match the PoC).
- [ ] T7 — If PASS: open a Plan in `katgpt-rs/.plans/` (open primitive) + a runtime guide in `riir-ai/.research/` (private selling point). If FAIL: leave G8 documented, close this issue with the negative-result record.

## Non-goals

- Real-model validation (Gemma 2 2B or Kimi K3) — deferred. The PoC is purely synthetic.
- Crowd-scale latency (N=1000+) — Set Attention G3 already documents the O(N²) scaling limit; this PoC stays at N=64.
- Chain commitment (LatCal / BLAKE3) — out of scope for a PoC.
- Multi-layer ablation — single-layer feedback payoff is sufficient to test the G8 hypothesis.

## Cross-references

- [Research 469](../.research/469_collective_intelligence_payoff_schemes.md) — the Gain verdict + paper distillation.
- [Research 255](../.research/255_VibeThinker_CLR_Test_Time_Reliability.md) — CLR (closest cousin for the feedback-payoff shape).
- [Research 274](../.research/274_Optimal_CCE_Moderator_LP_No_Regret.md) — CCE Moderator (designer-steerable crowd coordination).
- [Research 354](../.research/354_Cross_Datapoint_Set_Attention_NPT.md) — Set Attention (the primitive whose G8 limitation is the target).
- [Bench 354](../.benchmarks/354_set_attention_goat.md) L71 — the G8 documented failure.
- [Bench 284](../.benchmarks/284_clr_goat.md) — CLR G1 +78pp over majority (the operational evidence that feedback-payoff shape beats averaging).
- [riir-ai/.research/143](../../riir-ai/.research/143_Latent_CCE_Moderator_Crowd_Emergent_Coordination.md) — Latent CCE Moderator runtime guide (private crowd-scale CCE wiring).
