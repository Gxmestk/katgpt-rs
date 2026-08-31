# Plan 584: Abductive Repair Kernel — hodge-blame routing + hypothesis repair seam (Research 522 Track A open primitive)

**Status:** OPEN — Phase 0 not started; blocked on nothing (generic math, no consumer wiring inside this plan). The riir-ai consumer (Plan 560) is gated on PoC Issue 832 — this plan ships the generic kernel only.

**Source:** Research 522 (arXiv:2608.27549 Code-as-World) × riir-clippy heal-loop discipline. Generic, no game semantics — the game-side instantiation lives in riir-ai.

**Modelless mandate:** zero gradient steps; closed-form arithmetic only. Feature flag `abductive_repair` (opt-in) until GOAT; demote the loser if a simpler incumbent wins the same slot.

## Substrate consumed (substrate-first — nothing rebuilt)

| Need | Shipped substrate | Repo |
|---|---|---|
| Residual-field decomposition (blame routing) | `hodge_decompose`, `exterior_derivative`, `codifferential` | `katgpt-dec` (re-exported `katgpt_core::dec`) |
| Trajectory residuals / regime gates | `kinematics::residual_event`, `regime_predicates`, `extrapolation_horizon` | `katgpt-core::kinematics` (Plan 578) |
| Loop discipline (budget, decline, trajectory recording) | riir-clippy heal-loop shape: bounded fixpoint, first-class decline, EvolveRecorder trajectories | pattern only — no dep |
| Candidate ordering under noise (optional) | `beta_lcb_order_into`, `rating::expected_f32` | `katgpt-core` |

## Phase 0 — traits + blame routing

- [ ] `katgpt-core/src/abductive.rs` behind `#[cfg(feature = "abductive_repair")]`; module doc states the sync-boundary rule (outputs are latent-side repairs; nothing here touches sync surfaces).
- [ ] `trait WorldHypothesis: Clone + PartialEq` — components addressable as a fixed `SlotId` enum (`State`, `Param(u8)`, `Rule(u8)`); `fn simulate(&self, steps: u32, out: &mut [f32])` (caller-owned scratch, alloc-free); `fn refit_component(&mut self, slot: SlotId, residuals: &[f32]) -> RefitReport` (closed-form least squares over the slot's typed scalars).
- [ ] `enum Blame { State, Param(u8), Rule(u8), Structural }` + `fn hodge_blame(residual_field: &[f32], n: usize) -> Blame` — runs `dec::hodge_decompose` over the residual sequence; exact-energy dominance → `State`, coexact dominance → `Param/Rule` (by coexact spectral peak vs the slot's regime table), harmonic dominance → `Structural`. Pure function, zero alloc.
- [ ] Unit gates: planted-fault separability — scale error (expect exact-dominant), timing/phase error (coexact-dominant), wrong-body-count (harmonic-dominant); all three classified with zero ground-truth labels.

## Phase 1 — repair loop seam

- [ ] `struct RepairBudget { max_rounds: u32 }` + `enum RepairVerdict { Repaired { rounds: u32 }, Rejected { rounds: u32 } }` — REJECT is a first-class outcome, never a forced fit (decline falsifiability is a test, not prose).
- [ ] `fn repair<H: WorldHypothesis, V: Fn(&H, &[f32]) -> f32>(hyp: &mut H, observations: &[f32], verify: V, budget: RepairBudget) -> RepairVerdict` — per round: simulate → residual field → `hodge_blame` → local `refit_component` on the blamed slot only → verify; accept on sufficient ∧ parsimonious (verify score ≥ τ ∧ slot-diff count ≤ κ); reject at budget.
- [ ] Locality invariant gate: across N repair rounds, every non-blamed slot's bytes are bit-identical (the `zone_gating` frozen-gate invariant, pinned by test).
- [ ] Trajectory-recording hook (EvolveRecorder shape): each round records (round, blame, refit report, verify score) into a caller-provided sink — no global state.

## Phase 2 — falsifiable harness (in-repo, generic domain)

- [ ] `tests/abductive_repair_g1.rs`: deterministic double-run bit-identity; decline-falsifiability (structural fault → `Rejected`, never `Repaired` with a forced fit); matched-budget A/B — verify-guided repair vs independent re-sampling at equal simulation budget, repair arm must win on residual L2 in ≥ 4/5 fixture families (the paper's Fig. 4 ordering law as the pre-registered claim).
- [ ] GOAT gates: G1 determinism; G2 µs-class per repair round at 10⁴-element residual fields (release); G3 default-off until pass; G4 zero alloc in `repair` steady state (tracking allocator).

## Phase 3 — promotion decision

- [ ] If GOAT passes: keep `abductive_repair` opt-in (no default consumer inside katgpt-rs — promotion to default happens only when a consumer exists and demands it, per the no-default-consumer rule).
- [ ] Record honest negatives in Research 522 §10 if any gate fails; do not soften bars post-hoc.

## Non-goals

- No LLM in the loop (REx/LLM synthesis is 275/145 cold-tier territory; this kernel is the between-inductions runtime maintenance loop).
- No game types, no sync types, no archetype logic (riir-ai owns those in Plan 560).
- No video/appearance channel (Research 522 discard rows 14–15).
