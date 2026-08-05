# Issue 572: `katgpt-ruliology` GOAT Re-Gate (Loser-Sweep Category 1 PENDING)

**Filed:** 2026-08-05
**Origin:** Discussion of Wolfram's "Games between Programs: The Ruliology of
Competition" essay (June 2026). Research 168 already distills the essay at
GOAT-tier with 5 fusion primitives, and Plan 188 shipped
`katgpt-rs/crates/katgpt-ruliology/` (10 modules: `fsm`, `ca`, `tm`, `payoff`,
`bandit`, `mutation`, `irreducibility`, `simulation_gate`, `types`, `tests`).
**However**, the loser-sweep audit
(`katgpt-rs/.docs/10_audits/loser_sweep_audit.md`) lists `ruliology` under
**Category 1 — PENDING (stays in domain crate, GOAT not yet run)**, while
Research 168's verdict table claims all 5 fusions are "✅ GOAT, ✅ Default".
That is an architectural-claim-vs-empirical-evidence gap — exactly the §3.6
defend-wrong pattern: the GOAT was *claimed* from architectural coverage, not
*measured* against the shipped tests.

**Goal:** run the existing GOAT gate tests in
`katgpt-rs/crates/katgpt-ruliology/src/tests/` empirically, document the
results honestly (PASS or FAIL per gate), and update the audit + research
note + Cargo feature default to match reality. Promote to `default` if the
gate passes; keep opt-in (or demote) if it fails.

**Scope:** this is a proof/audit task — run tests, document, update docs.
No new primitives. If a gate is missing, write the minimal test that closes
the gap. If a gate fails, document the failure mode and create a follow-up
issue; do NOT silently lower the bar.

## Context — what already ships

The crate's test suite is more complete than the loser-sweep audit suggests:

| File | Contents |
|---|---|
| `tests/wolfram_results.rs` | 13 G1 correctness tests — verifies Wolfram's published numbers (22 distinct 2-state FSMs, grim trigger > tit-for-tat in PD, complexity-payoff correlation ≈ 0, always-defect exploits always-cooperate, cross-paradigm FSM vs CA vs TM) |
| `tests/benchmarks.rs` | 9 G2 perf tests — enumeration time (N=2/3/4), tournament time, IrreducibilityGate overhead (sub-µs target), no-regression placeholder |

The G2 tests already encode Wolfram-derived perf budgets:
- FSM(2) enumerate < 500ms
- FSM(3) enumerate < 10s
- CA enumerate < 100ms (all) / < 1000ms (distinct)
- TM enumerate < 50ms
- FSM(2) tournament < 500ms
- FSM(3) tournament < 60s
- IrreducibilityGate < 1ms/check (FSM2) / < 100ms/check (FSM3)

## GOAT gate mapping (Research 168's 5 fusions)

| Fusion | Claimed | Test coverage | Gate to run |
|---|---|---|---|
| F1 RuliologyBandit | ✅ GOAT, ✅ Default | enumeration + tournament benches | `bench_enumerate_fsm_*`, `bench_tournament_fsm_*` |
| F2 CrossParadigmArena | ✅ GOAT, ✅ Default (test/example) | `test_cross_paradigm_*` (4 tests) | run wolfram_results.rs |
| F3 IrreducibilityGate | ✅ GOAT, ✅ Default (gate) | `bench_irreducibility_gate_*` | run benchmarks.rs |
| F4 RuliologyPruner | ✅ GOAT, ✅ Default | Pareto-front filter — verify test coverage | grep for `RuliologyPruner` tests |
| F5 AdaptiveStrategyMutation | ✅ GOAT, 🔧 Feature-gated | `mutation::co_evolve`, `delta_gated_co_evolve` (feature-gated) | run with `--features ruliology` |

## Tasks

- [x] **T1** Run `cargo test -p katgpt-ruliology --features ruliology --lib` and capture full output — 98 tests (96 passed, 1 failed FSM(3) debug-mode perf, 1 ignored FSM(4))
- [x] **T2** Tally G1 (correctness — wolfram_results.rs) pass/fail per test — 13/13 PASS
- [x] **T3** Tally G2 (perf — benchmarks.rs) pass/fail per test — 8/8 PASS (release); FSM(3) debug-mode failure resolved via `#[cfg_attr(debug_assertions, ignore)]` (18.8s debug vs 294ms release = 64× ratio)
- [x] **T4** Verify F4 (RuliologyPruner) test coverage — covered by `types::tests::test_ruliology_pruner_filter`, `test_pareto_front_filters_dominated`, `test_pareto_front_complexity_collision_returns_correct_ids`
- [x] **T5** Verify F5 (AdaptiveStrategyMutation) test coverage — covered by `mutation::tests::test_co_evolve_converges`, `test_delta_gated_*` (3 tests), `test_propose_*` (3 tests)
- [x] **T6** Run `cargo clippy -p katgpt-ruliology --all-targets --features ruliology` — clean, no warnings
- [x] **T7** Document verdict: GOAT gate PASSES. Loser-sweep audit updated (Cat-1 → annotated PASS). Research 168 verdict table corrected (overclaim "✅ Default" → "🔧 opt-in"). Feature stays opt-in (niche tool, pulls `bandit` dep).
- [-] **T8** N/A — no gate failed
- [x] **T9** Benchmark note written: [`.benchmarks/572_katgpt_ruliology_goat.md`](../.benchmarks/572_katgpt_ruliology_goat.md)
- [x] **T10** Commit pending

## Promotion criteria (per AGENTS.md GOAT gate rule + §1.55)

- **G1 correctness**: all wolfram_results.rs tests pass (Wolfram's numbers reproduce)
- **G2 perf**: all benchmarks.rs tests pass within the encoded budgets
- **G3 no-regression**: `cargo clippy` clean; `cargo test --lib` (root, without feature) still passes
- **G4 alloc-free or equivalent**: not applicable — ruliology runs offline (enumeration is one-shot at boot, not per-tick); the analog is "enumeration completes within budget" which G2 covers

The gate is **modelless** (no training, no backprop — FSM enumeration is deterministic combinatorics). Per AGENTS.md: "Promotion requires modelless gain." This is satisfied by construction.

## Non-goals

- Unshelving Proposal 012 (Living Dungeon × NCA × DEC-as-CA-on-terrain) — that is a separate game-content plan, not a GOAT re-gate. The *Stars Reach*-style living-terrain angle stays out of scope here.
- Adding new ruliology primitives. If the existing 5 fusions pass, that's the gate. New fusions go through Research → Plan, not this issue.
- Cross-repo propagation (riir-ai consumer wiring). The `riir-games/ruliology/` consumer module exists; verifying its wiring is a separate goat-audit task.

## References

- [Research 168](../.research/168_Ruliology_Competition_Enumerative_Game_Theory.md) — original GOAT-tier distillation
- [Plan 188](../.plans/188_*.md) (if exists) — original implementation plan
- [Loser-sweep audit](../.docs/10_audits/loser_sweep_audit.md) — Category 1 PENDING list
- `katgpt-rs/crates/katgpt-ruliology/README.md` — crate overview
