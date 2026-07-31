# Plan 456: Latent Error Diffusion — Defend-Wrong POC

**Date:** 2026-07-17
**Research:** [katgpt-rs/.research/448_Latent_Error_Diffusion_Dual_Stream.md](../.research/448_Latent_Error_Diffusion_Dual_Stream.md)
**Source paper:** [arxiv 2606.31700](https://arxiv.org/abs/2606.31700) — Yamada et al., *Diffusing Blame: Task-Dependent Credit Assignment in Biologically Plausible Dual-Stream Networks* (Sakana AI, 30 Jun 2026)
**Target:** `riir-ai/crates/riir-poc/` (defend-wrong PoC per research skill §3.6)
**Status:** Phase 1 DONE (scaffold + harness); Phase 2 DONE (Option C refactor + G1–G6 mechanism gates all PASS); Phase 3 DONE (G7 FAILS, G8–G10 PASS) → Phase 4 verdict: Research 448 revised to **Pass** (modelless-correct but task-useless; honest negative result).

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

- [x] **T1.1** Create `riir-ai/crates/riir-poc/src/latent_ed_poc.rs` — three competitors + toy domain + verdict printer
- [x] **T1.2** Implement `LatentEdState` — dual-stream `(p, n)` latent state, four non-negative projection matrices `W_pp, W_np, W_nn, W_pn`, modulo routing matrix `M`, layer-specific sigmoid width α
- [x] **T1.3** Implement `LatentEdState::step()` — forward pass (dual-stream sigmoid update) + action selection (argmax, never softmax per AGENTS.md) + ED update (local Hebbian-style, no backprop)
- [x] **T1.4** Implement `FrozenBaseline` — same dual-stream state, same forward pass, **no ED update** (pure projection). This is the no-adaptation control.
- [x] **T1.5** Implement `TilrCompetitor` — wrap Plan 425's `tilr_refine_into` as the runtime belief-update mechanism (the closest shipped cousin)
- [x] **T1.6** Implement `toy_decision_task(seed)` — K=10 action decision task with controlled nonlinear reward (matches paper's 10-way MNIST/CIFAR setup). Reward = `sin(action · latent · direction) + noise`. 1000-step horizon.
- [x] **T1.7** Implement `verdict_table(seeds)` — runs all 3 competitors × 5 seeds × 1000 steps, prints accuracy / latency / variance table
- [x] **T1.8** Add `latent_ed_poc` to `riir-poc/Cargo.toml` `[[bench]]` section + `src/lib.rs` re-export

**Verification:** `cargo bench -p riir-poc --bench latent_ed_poc -- --nocapture` prints the verdict table. ✅ Verified — bench runs, prints the table, computes G7–G10.

### Phase 1 preliminary finding (recorded honestly per §3.6)

The scaffold runs end-to-end on all three competitors. The verdict table
prints. **However, G7 FAILS decisively in a revealing way**: Latent-ED's
accuracy is **bit-identical** to the Frozen baseline (0.5322 vs 0.5322,
same seed variance) across all 5 seeds. The ED update is having **zero
effect** on chosen actions.

**Root cause analysis (preliminary):** The forward pass fully overwrites
`(p, n)` with sigmoid outputs every tick:
```rust
self.p[h] = sigmoid((p·W_pp − n·W_np + x) / α_p)
```
The tiny ED delta (`dp ≈ η·p·σ'(Z)·R_h ≈ 0.0006` per tick) is far below
the forward pass's noise floor — next tick's `temp_p = p·W_pp` is dominated
by the W_pp matrix values (`~0.375 per entry × 64 entries ≈ 12`), so
sigmoid(12/6) ≈ 0.88 regardless of the ED contribution. The ED update is
washed out by the recurrent dynamics.

**This is a reframing bug, not a paper refutation.** The paper's ED rule
modifies **weights** (which persist across forward passes). My latent
translation modifies the **activations** (which get overwritten by the
next forward pass). The correct latent reframing is one of:

- **Option A (additive bias):** `(p, n)` is a persistent bias added to
  the forward pass, not overwritten by it. `p_new = sigmoid(...) +
  accumulated_ed_delta`.
- **Option B (separate belief state):** ED modifies a separate
  `belief_state` that gates action selection directly, bypassing the
  recurrent dynamics.
- **Option C (state-only):** Drop the recurrent forward entirely. `(p, n)`
  IS the belief state, evolved only by the ED rule. W_* projections
  become a one-time input→initial-state transform.

**Phase 2 must pick one option and re-implement before the mechanism gates
G1–G6 are meaningful.** The current scaffold proves the evaluation harness
works (the table renders, the gates compute, the smoke test passes); the
mechanism itself needs the reframing fix.

Notably, TILR (which DOES update a persistent state via `tilr_refine_apply`)
achieves 47.3% accuracy — worse than Frozen's 53.2%. This is because TILR's
invariant-subspace projection (rank 4) is too restrictive for this toy
task's signal structure. The headline gate G8's current PASS (ED beats TILR
by 5.9pp) is therefore **vacuous** — both ED and TILR are broken in
*different* ways. After the Phase 2 reframing, G7–G10 must be re-run.

**Verdict: Phase 1 is delivered (scaffold + harness work). Phase 2 (the
mechanism gates) is BLOCKED on the reframing decision (Option A/B/C).**
The user should decide which option to pursue before Phase 2 starts.

---

## Phase 2 — Mechanism-Level Gates (modelless correctness)

### Option C Refactor (prerequisite, landed 2026-07-18)

Per user decision (2026-07-18): Option C — make `(p, n)` persist across
steps instead of being recomputed. The Phase 1 reframing bug was that the
recurrent forward pass overwrote the ED delta each tick. Option C resolves
this by construction: drop the recurrent forward entirely; `(p, n)` is
the persistent belief state, evolved only by the ED rule; `W_*`
projections become a one-time input→initial-state transform (applied
lazily on the first `forward()` call after `reset()`).

**Changes landed** (`riir-ai/crates/riir-poc/src/latent_ed_poc.rs`):

- Removed unused state fields `z_p, z_n, sigma_p, sigma_n, temp_p, temp_n`
  (no recurrent forward → no pre-activation cache needed).
- Added `bootstrapped: bool` flag controlling the one-time W_* transform.
- `forward()` is now **read-only** over `(p, n)` post-bootstrap: it only
  selects actions via argmax against the committed direction vectors.
- New private `bootstrap()` method runs the W_* projection once using
  stack-local `[f32; H]` scratch (zero heap allocation).
- `apply_ed_update()` computes the drive gate as `p * (1 − p)` directly
  (treating the bounded `p` AS the σ output, which it is by construction).
- Added two new invariant tests: `latent_ed_forward_is_read_only_post_bootstrap`
  and `frozen_baseline_matches_latent_ed_pre_update`.

### Tasks

- [x] **T2.1** **G1 — Update is local (no backprop)** — assert steady-state `forward()` + `apply_ed_update()` allocate 0 bytes (verified via Vec capacity stability across HORIZON=1000 steps; CountingAllocator unavailable because `katgpt-rs` dep claims `#[global_allocator]`, so capacity stability + structural inspection form the proof). **PASS.**
- [x] **T2.2** **G2 — No weight mutation** — BLAKE3 hash of the four `W_*` matrices is bit-identical before and after 1000 `step()` calls. **PASS.**
- [x] **T2.3** **G3 — Sigmoid bounded** — all entries of `(p, n)` stay in `[0, 1]` after every step (spot-checked every 100 ticks across HORIZON, including adversarial saturating error magnitudes). **PASS.**
- [x] **T2.4** **G4 — Argmax, not softmax** — action selection returns binary `a ∈ {0, 1}` across 200 steps × 10 channels; both values appear (no stuck-at degeneracy). **PASS.**
- [x] **T2.5** **G5 — Modulo routing is deterministic** — routing identity `R_h = error[h % C]` verified for all (h, h') pairs sharing a channel. **PASS** (holds by construction — modulo is a pure function).
- [x] **T2.6** **G6 — E/I balance tracking** — `‖p‖ / ‖n‖` stays in `[0.5, 2.0]` sampled every 50 ticks across HORIZON; recorded balance history (every 100 steps) all in range. **PASS.**
- [x] **T2.7** Integration harness: all three competitors complete a full run without panic (`full_run_harness_all_three_competitors_complete`). **PASS.**

**Verification:** `cargo test -p riir-poc --test latent_ed_mechanism_gates`
→ 7 passed; 0 failed. ✅ Verified.

**Mechanism verdict:** G1–G6 ALL PASS. The latent-ED rule is modelless
and correct at the mechanism level — zero-alloc hot path, no weight
mutation, bounded state, binary argmax, deterministic modulo routing,
E/I balance stable. This says nothing about whether the rule is *useful*
(Phase 3 settles that).

---

## Phase 3 — Quality Gate (the defend-wrong race)

### Tasks

- [x] **T3.1** **G7 — Accuracy vs Frozen baseline** — **FAIL ❌**. Latent-ED 0.5312 vs Frozen 0.5363 (−0.51 pp). The Option C refactor fixed the Phase 1 "ED has zero effect" bug (the two competitors are no longer bit-identical), but the ED rule is actively *hurting* accuracy by a small margin — the Hebbian update accumulates in a direction that does not align with the task's reward structure.
- [x] **T3.2** **G8 — Accuracy vs TILR (the headline gate)** — **PASS ✅ (vacuous)**. Latent-ED 0.5312 vs TILR 0.4730 (+5.83 pp). **But this is vacuous**: TILR is broken on this toy task (47.30% < Frozen's 53.63% — TILR's rank-4 invariant-subspace projection is too restrictive for the task's signal structure). Beating a broken competitor proves nothing.
- [x] **T3.3** **G9 — Variance check** — **PASS ✅**. Latent-ED seed-variance 0.00032 vs TILR 0.00456 (ratio 0.07×, well under the 1.5× threshold). The paper reported ED-PPO has higher variance than BP-PPO; the latent analog does NOT inherit this.
- [x] **T3.4** **G10 — Stability over horizon** — **PASS ✅**. Late accuracy (ticks 900–1000) = 0.5228 ≥ early accuracy (ticks 100–200) = 0.5022. No catastrophic forgetting / divergence over the horizon. The ED-driven state evolution is stable, just not *useful*.
- [x] **T3.5** Run all gates with `CARGO_TARGET_DIR=/tmp/latent_ed_poc_phase2` per AGENTS.md, clean up when done.

**Verification:** verdict table from T1.7 + G7–G10 assertions. Printed honestly — G7 FAIL, G8–G10 PASS. ✅ Verified (bench re-run confirms stable numbers).

### Phase 3 honest finding

The Option C refactor **succeeded mechanically** (ED now has measurable
effect; Phase 1's bit-identical-to-Frozen bug is gone) but **failed
empirically** (the effect is slightly negative: −0.51 pp vs Frozen). The
latent-ED rule, even with correct persistence semantics, does not produce
a useful belief update on this K=10 toy task. The local Hebbian update
`Δp ∝ p · σ'(Z) · R_h` accumulates, but accumulates in a direction that
*reduces* accuracy relative to the fixed bootstrap state.

**This is an honest negative result per §3.6.** The mechanism is
modelless-correct (G1–G6 all PASS) but task-useless (G7 FAILS). The
defend-wrong PoC has done its job: it defended the hypothesis as hard
as the modelless reframing allows (Option C is the most faithful
semantics), and the hypothesis failed the quality bar.

Notably, TILR — the *other* belief-update competitor — also underperforms
Frozen (47.30% vs 53.63%). This suggests the toy K=10 task may not be a
favorable domain for *any* runtime belief-update mechanism: the reward
signal structure rewards the fixed bootstrap projection more than any
error-driven refinement. This is a property of the task, not the ED rule
specifically. A follow-up plan could probe whether a different task (e.g.
K=4 game-action stretch per T5.1) shows a different ranking — but that's
out of scope for this PoC's verdict.

---

## Phase 4 — Honest Verdict + Routing (POST-POC)

### Tasks

- [x] **T4.1** Wrote verdict addendum to `katgpt-rs/.research/448_*.md` §"PoC Addendum" with raw numbers (accuracy / latency / variance for all 3 competitors). Honest per §3.6 — verdict recorded as **modelless-correct but task-useless**.
- [x] **T4.2** G7–G10 did NOT all PASS (G7 FAILS) → **no follow-up plan opens in katgpt-rs**. The `latent_error_diffusion` feature flag candidate is **dropped**.
- [x] **T4.3** N/A (G7 FAILS, not just G8).
- [x] **T4.4** N/A (G9 PASSES — no ×TILR fusion needed).
- [x] **T4.5** The mechanism is NOT broken at the modelless level (G1–G6 all PASS), but G7 FAILS (the rule is task-useless, not mechanism-broken). Per the honest verdict routing: **Research 448 revised down to Pass** (negative result recorded). The PoC is kept as a regression check in riir-poc — the mechanism gates (G1–G6) remain valuable as a documented modelless-correctness proof even though the rule is not promoted.

### Final Verdict

**Research 448: Pass (negative result).** The latent-ED rule (Yamada et
al. 2026, arxiv 2606.31700) is **modelless-correct** (G1–G6 all PASS:
local Hebbian update, no weight mutation, bounded state, argmax action
selection, deterministic modulo routing, E/I balance stable) but
**task-useless** (G7 FAILS: −0.51 pp vs the Frozen baseline on the K=10
toy decision task). The Option C persistent-belief-state reframing — the
most faithful modelless translation — does not unlock an accuracy gain.

The PoC is retained in `riir-poc` as a regression check. No katgpt-rs
primitive ships. No feature flag is created.

---

## Phase 5 — Stretch Goals (BLOCKED — Phase 3 G7 failure)

**Status: NOT PURSUED.** Phase 3's G7 failure (Latent-ED −0.51 pp vs Frozen)
blocks the stretch goals by definition — there's no GOAT-grade result to
extend. The stretch goals are preserved below for historical reference; a
future plan could reopen them if a different task domain shows Latent-ED
beating Frozen (the K=10 toy task is an unfavorable domain for runtime
belief updates in general — TILR also underperforms Frozen here).

### Tasks (NOT PURSUED)

- [-] **T5.1** **K=4 game-action stretch** — would test fine-grained temporal credit assignment. Blocked by G7 failure on K=10. Reopen only if a K=10 follow-up shows a different ranking.
- [-] **T5.2** **×TILR fusion PoC** — G9 PASSES (variance is fine), so no mandatory fusion trigger. Not pursued.
- [-] **T5.3** **LatCal commitment stretch** — gated on a working accuracy mechanism; not applicable to a task-useless rule.

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

Defend-wrong PoC for the latent-ED hypothesis, **settled (negative result)**.
Three competitors (Latent-ED vs Frozen vs TILR) raced on a K=10
multi-action decision task. G1–G6 PASS (mechanism is modelless-correct).
G7 FAILS (−0.51 pp vs Frozen — the rule is task-useless). G8 PASSES but
vacuously (TILR is also broken on this task). G9–G10 PASS. Research 448
revised to **Pass** (negative result). The Option C persistent-belief-state
reframing — the most faithful modelless translation per user decision —
does not unlock an accuracy gain. No katgpt-rs primitive ships. PoC kept
as a regression check.
