# Plan 456: Latent Error Diffusion — Defend-Wrong POC

**Date:** 2026-07-17
**Research:** [katgpt-rs/.research/448_Latent_Error_Diffusion_Dual_Stream.md](../.research/448_Latent_Error_Diffusion_Dual_Stream.md)
**Source paper:** [arxiv 2606.31700](https://arxiv.org/abs/2606.31700) — Yamada et al., *Diffusing Blame: Task-Dependent Credit Assignment in Biologically Plausible Dual-Stream Networks* (Sakana AI, 30 Jun 2026)
**Target:** `riir-ai/crates/riir-poc/` (defend-wrong PoC per research skill §3.6)
**Status:** Active — Phase 1 (POC scaffold)

---

## Goal

Settle the user's hypothesis ("latent-ED gains on accuracy over existing belief-update primitives") with a **defend-wrong PoC** in `riir-ai/crates/riir-poc/`. The PoC's job is to defend OR refute — both outcomes are valid per §3.6.

Three competitors race on a controlled toy multi-action decision task:
1. **Latent-ED** (paper's mechanism, modellessly translated per Research 448 §2.1)
2. **Frozen baseline** (no belief update — pure direction-vector projection)
3. **TILR refinement** (Plan 425 — closest shipped cousin, invariant-subspace correction)

The PoC prints a verdict table. If Latent-ED beats TILR by ≥5 pp accuracy at ≤2× latency → GOAT, promote. Else → Research 448 stays Gain (opt-in feature, no default promotion).

**No katgpt-rs primitive ships in this plan.** This is a POC-only plan. If the POC PASSes, a follow-up plan (in katgpt-rs) ships `katgpt-core/src/sense/latent_ed.rs` behind feature `latent_error_diffusion`.

---

## Phase 1 — POC Scaffold (CORE)

### Tasks

- [ ] **T1.1** Create `riir-ai/crates/riir-poc/src/latent_ed_poc.rs` — three competitors + toy domain + verdict printer
- [ ] **T1.2** Implement `LatentEdState` — dual-stream `(p, n)` latent state, four non-negative projection matrices `W_pp, W_np, W_nn, W_pn`, modulo routing matrix `M`, layer-specific sigmoid width α
- [ ] **T1.3** Implement `LatentEdState::step()` — forward pass (dual-stream sigmoid update) + action selection (argmax, never softmax per AGENTS.md) + ED update (local Hebbian-style, no backprop)
- [ ] **T1.4** Implement `FrozenBaseline` — same dual-stream state, same forward pass, **no ED update** (pure projection). This is the no-adaptation control.
- [ ] **T1.5** Implement `TilrCompetitor` — wrap Plan 425's `tilr_refine_into` as the runtime belief-update mechanism (the closest shipped cousin)
- [ ] **T1.6** Implement `toy_decision_task(seed)` — K=10 action decision task with controlled nonlinear reward (matches paper's 10-way MNIST/CIFAR setup). Reward = `sin(action · latent · direction) + noise`. 1000-step horizon.
- [ ] **T1.7** Implement `verdict_table(seeds)` — runs all 3 competitors × 5 seeds × 1000 steps, prints accuracy / latency / variance table
- [ ] **T1.8** Add `latent_ed_poc` to `riir-poc/Cargo.toml` `[[bench]]` section + `src/lib.rs` re-export

**Verification:** `cargo bench -p riir-poc --bench latent_ed_poc -- --nocapture` prints the verdict table.

---

## Phase 2 — Mechanism-Level Gates (modelless correctness)

### Tasks

- [ ] **T2.1** **G1 — Update is local (no backprop)** — assert `LatentEdState::step()` allocates 0 bytes (CountingAllocator) and contains no autograd / reverse-mode AD. The update rule is `Δp += η · p_old · U_p` etc. — pure forward arithmetic.
- [ ] **T2.2** **G2 — No weight mutation** — assert the four `W_*` projection matrices are bit-identical before and after 1000 `step()` calls. Only `(p, n)` latent state mutates.
- [ ] **T2.3** **G3 — Sigmoid bounded** — assert all entries of `(p, n)` stay in `[0, 1]` after every step (wide sigmoid + Dale's sign routing bounds the state by construction).
- [ ] **T2.4** **G4 — Argmax, not softmax** — assert action selection uses `argmax_c (p · d_c − n · d_c)`, never softmax (AGENTS.md mandate).
- [ ] **T2.5** **G5 — Modulo routing is deterministic** — assert `M[h,c] = (h mod C == c)` and that the routing matrix is fixed across all steps (not learned).
- [ ] **T2.6** **G6 — E/I balance tracking** — log `||p|| / ||n||` at every 100 steps; assert it stays in `[0.5, 2.0]` (no collapse to all-excitatory or all-inhibitory). If violated → record as honest failure mode per §3.6.

**Verification:** `cargo test -p riir-poc --test latent_ed_mechanism_gates`

---

## Phase 3 — Quality Gate (the defend-wrong race)

### Tasks

- [ ] **T3.1** **G7 — Accuracy vs Frozen baseline** — Latent-ED accuracy ≥ Frozen baseline accuracy + 10 pp (the no-adaptation control should be easy to beat; if not, the ED rule is broken)
- [ ] **T3.2** **G8 — Accuracy vs TILR (the headline gate)** — Latent-ED accuracy ≥ TILR accuracy + 5 pp **OR** Latent-ED latency ≤ TILR latency × 0.5 at parity accuracy. Either branch is a GOAT-grade result; both failing is a Gain-only result.
- [ ] **T3.3** **G9 — Variance check** — Latent-ED seed-variance ≤ 1.5 × TILR seed-variance. Paper reports ED-PPO has higher variance than BP-PPO; the latent analog should NOT inherit this. If it does → mandatory ×TILR fusion (use TILR's invariant-subspace gate as the postsynaptic drive gate) before any promotion.
- [ ] **T3.4** **G10 — Stability over horizon** — Latent-ED accuracy in ticks [900..1000] ≥ accuracy in ticks [100..200] (no catastrophic forgetting / divergence over the horizon).
- [ ] **T3.5** Run all gates with `CARGO_TARGET_DIR=/tmp/latent_ed_poc` per AGENTS.md, clean up when done.

**Verification:** verdict table from T1.7 + G7–G10 assertions. Print honestly — PASS or FAIL with the numbers.

---

## Phase 4 — Honest Verdict + Routing (POST-POC)

### Tasks

- [ ] **T4.1** Write verdict addendum to `katgpt-rs/.research/448_*.md` §"PoC Addendum" with raw numbers (accuracy / latency / variance for all 3 competitors). Honest per §3.6 — do NOT silently revise the verdict to match a hoped-for outcome.
- [ ] **T4.2** If G7–G10 all PASS → open follow-up plan in katgpt-rs to ship `latent_error_diffusion` feature in `katgpt-core/src/sense/`. Promote to default-on after that plan's GOAT gate.
- [ ] **T4.3** If G8 FAILS (no accuracy gain over TILR) but G7 PASSes (mechanism works) → Research 448 stays **Gain**. Keep POC as regression check in riir-poc. Do NOT promote.
- [ ] **T4.4** If G9 FAILS (variance too high) → test the ×TILR fusion (use TILR's invariant-subspace projection as the postsynaptic drive gate in the ED rule). Re-run G7–G10 with the fused variant.
- [ ] **T4.5** If the entire mechanism is broken (G1–G6 fail) → honestly revise Research 448 verdict down to **Pass** (mechanism doesn't translate to latent space). Delete the feature flag candidate. Record the negative result.

---

## Phase 5 — Stretch Goals (only if Phase 3 PASSes)

### Tasks

- [ ] **T5.1** **K=4 game-action stretch** — re-run G7–G10 with K=4 actions (fight/flee/forage/talk) on a 100-tick toy NPC sim. Tests the "Craftax shortfall class" — does fine-grained temporal credit assignment work, or does coarse modulo routing fail?
- [ ] **T5.2** **×TILR fusion PoC** — even if G9 PASSes, test the ×TILR fusion (TILR invariant-subspace gate as ED postsynaptic drive gate). If the fusion strictly dominates vanilla Latent-ED → that's the Super-GOAT re-open trigger per Research 448 §3.
- [ ] **T5.3** **LatCal commitment stretch** — project `(p, n)` to the 5 synced affect scalars (valence/arousal/desperation/calm/fear) via bridge function. Assert the bridge is zero-allocation, gateable, and the synced raw values are bit-identical across two runs (deterministic replay check).

---

## Failure-mode prophylactics (the honest caveats)

These are the paper's reported failure modes; the PoC must explicitly probe each:

| Paper failure mode | PoC probe | Mitigation if observed |
|---|---|---|
| ED-PPO higher variance than BP-PPO across all envs | G9 (variance check) | ×TILR fusion (Phase 5 T5.2) or reject |
| Craftax shortfall (−4.0 reward, fine-grained temporal credit assignment fails) | T5.1 (K=4 stretch) | Limit promotion to coarse-action domains; document honestly |
| 25× surrogate gradient attenuation (sigmoid vanishing) | G3 (bounded state) — wide α prevents this | Already mitigated by α=3–6 |
| E/I balance divergence | G6 (balance tracking) | Reject if divergence observed (mechanism broken) |
| Implicit sparsity loss of capacity | Not gated — emergent property, monitor only | If capacity loss is severe, add floor-lift warm restart |

---

## Build / Run

```bash
# All PoC work uses isolated target dir per AGENTS.md
export CARGO_TARGET_DIR=/tmp/latent_ed_poc

# Run the verdict table (Phase 1)
cargo bench -p riir-poc --bench latent_ed_poc -- --nocapture

# Run the mechanism-level gates (Phase 2)
cargo test -p riir-poc --test latent_ed_mechanism_gates

# Clean up when done
rm -rf /tmp/latent_ed_poc
```

---

## References

- [Research 448](../.research/448_Latent_Error_Diffusion_Dual_Stream.md) — the parent research note
- [Research 408](../.research/408_Trajectory_Invariant_Latent_Refinement.md) + [Plan 425](425_tilr_invariant_subspace_refinement.md) — TILR (closest cousin + Phase 3 competitor)
- [Research 236](../.research/236_QGF_Test_Time_Q_Guided_Flow.md) + [Plan 268](268_qgf_test_time_q_guided_flow.md) — QGF (flow analog, not a direct competitor but informs fusion)
- `.benchmarks/033_lt2_looped_goat.md` — AHLA recurrent substrate (the latent state substrate)
- Research skill §3.6 — defend-wrong PoC methodology (the protocol this plan follows)

---

## TL;DR

Defend-wrong PoC for the latent-ED hypothesis. Three competitors (Latent-ED vs Frozen vs TILR) race on a K=10 multi-action decision task. G1–G6 prove the mechanism is modelless and correct. G7–G10 are the quality race — G8 (vs TILR) is the headline gate. Honest verdict recorded in Research 448 §"PoC Addendum" regardless of outcome. No katgpt-rs primitive ships in this plan; follow-up plan opens only if G7–G10 PASS.
