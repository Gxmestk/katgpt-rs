# Plan 573: MOP Value-Iteration Primitive — `katgpt-core/src/mop/`

**Date:** 2026-08-15
**Research:** [katgpt-rs Research 478 — Maximum Occupancy Principle (Super-GOAT)](../.research/478_MOP_Maximum_Occupancy_Principle.md) · private half: [riir-ai Research 338 — runtime guide](../../riir-ai/.research/338_per_npc_mop_runtime_guide.md)
**Source paper:** [arXiv:2205.10316](https://arxiv.org/abs/2205.10316) — Ramírez-Ruiz, Grytskyy, Mastrogiuseppe, Habib, Moreno-Bote, *Complex behavior from intrinsic motivation to occupy future action-state path space*, Nat. Commun. 15, 6368 (2024). CC-BY 4.0.
**Target:** `crates/katgpt-core/src/mop/` (new module: `mod.rs` + `solve.rs` + `types.rs`) + Cargo feature `mop_path_entropy` (opt-in) + bench `benches/bench_mop_solver.rs`
**Status:** Active — Phase 0 (reference implementation proven in `riir-ai/crates/riir-poc/src/mop_poc.rs`, Bench 677; this plan productionizes it as a public primitive)

---

## Goal

Ship the generic `MopSolver<N, A>` value-iteration operator — the paper's Eq. 7 fixed-point map over a frozen tabular transition kernel — as a public, leaf-clean, modelless katgpt-core primitive. Pure math: no game semantics, no chain, no shard, no emotion vocabulary. This is Research 478 §3.3 mandatory output #1 (open primitive) and the P0 dependency for [riir-ai Plan 538](../../riir-ai/.plans/538_per_npc_mop_runtime.md) (the private runtime wiring).

**IP note (hard rule):** implement fresh from the paper's math (CC-BY 4.0). Do NOT copy `riir-poc` code — that file is private riir-ai IP. The PoC serves as the *parity oracle* consumed by riir-ai's Plan 538 T1.2 (which lives where both are visible), never as a source template here.

---

## Phase 1 — Primitive

### Tasks

- [ ] **T1.1** `types.rs`: `MopConfig { alpha: f32, beta: f32, gamma: f32, tol: f32, max_iter: u32 }` + `MopSolution<const N: usize, const A: usize> { v_star: [f32; N], log_z: [f32; N], iterations: u32, sup_delta: f32 }`. Constructor validates `alpha > 0`, `beta ≥ 0`, `gamma ∈ (0,1)`.
- [ ] **T1.2** `solve.rs`: `MopSolver<N, A>` — Eq. 7 in **log-space LSE form** (the PoC-proven formulation): `ln z_next[i] = γ · LSE_k( H̄[i,k] + γ·Σ_j p[i,k,j]·ln z[j] )` over available actions, with absorbing-state pinning (`ln z = 0` ⇒ `V = 0` exactly). Caller provides the kernel `&[[ [f32; N]; A]; N]`, mask `&[[u8; A]; N]`, config, and **scratch buffers** (no internal allocation). Convergence: sup-norm on `ln z` deltas vs `tol`, cap `max_iter`.
- [ ] **T1.3** Policy extraction: `pi_star(&solution, s, out: &mut [f32; A])` — closed-form from the fixed point. **Normalizer is `z^{1/γ}`** (the PoC's correction of Research 478 §2.1's pseudocode — Bench 677 deviation #4; do not "simplify" to `z^{-1}`).
- [ ] **T1.4** Entropy helpers: consume `crate::cgsp::types::entropy_nats` for `H(A|s)` (uniform over available) and add `state_conditional_entropy(p_row) -> f32` (`H(S'|s,a)` = −Σ p·ln p, 0-skip for exact zeros). No duplicate entropy code.
- [ ] **T1.5** Feature gate `mop_path_entropy = []` in katgpt-core `[features]` (opt-in; promotion only after the GOAT gate below passes modellessly). Module doc carries the selling-point one-liner + pointers to Research 478 / the private guide.

## Phase 2 — Tests (G1 correctness)

### Tasks

- [ ] **T2.1** Golden-parity test: a test-local reference implementation of paper Eq. 7 (straightforward non-LSE form, ~40 LOC — deliberately structurally different from `solve.rs`) on the paper Fig. 2a 4-room gridworld (N=82 incl. DEAD absorbing, A=4, 2 traps, 2 food, door gaps adjacent to center). Assert `V*` matches ≤ 1e-6 and `V(s+) = 0` exactly. **Honest deviation from Research 478 §3.3's original G1 wording** ("bit-identical to paper Eq. 7 reference Python implementation"): cross-language f32-vs-f64 bit-identity is unachievable by construction — the PoC (Bench 677) itself gated on tolerances. The achievable rigorous form: structurally-different same-precision reference + ≤1e-6 + the *exact* invariants (V(s+)=0 bit-exact, Theorem-3 init-invariance 0.0). Research 478 §3.3 item 1 updated to match (2026-08-15).
- [ ] **T2.2** Invariant battery (mirror the PoC's): π\* sums to 1 over available actions (≤1e-5), 0 on unavailable; Theorem-3 init-invariance (ones vs twos init → identical `V*`); convergence within `max_iter` on both arenas; ring arena (N=17, A=3) second domain.
- [ ] **T2.3** Edge cases: single-action states (H(A|s)=0), deterministic kernels (H(S'|s,a)=0 ⇒ β term vanishes), γ → 1 stability (tolerance-scaled), all-unavailable state (returns terminal).

## Phase 3 — Bench + GOAT gate

### Tasks

- [ ] **T3.1** Bench `benches/bench_mop_solver.rs` (feature-gated): full solve at (N,A) ∈ {(64,8), (64,16), (256,16)} + per-iteration cost; report µs/solve.
- [ ] **T3.2** **G2:** full solve < 1 ms at N=256/A=16 on M3 Max (the PoC hit sub-ms at N=82 with 290 iters; log-space matvec form should hold the margin — if not, record honestly).
- [ ] **T3.3** **G4:** CountingAllocator test — 0 allocations across a full solve + 1000 `pi_star` extractions (scratch provided by caller).
- [ ] **T3.4** **G3:** `cargo test -p katgpt-core --lib` unchanged with feature off; `--features mop_path_entropy` adds the new tests; `cargo clippy` clean both states; `--all-features` combo clean.
- [ ] **T3.5** **UQ floor ("Report the Floor"):** N/A with justification recorded in the bench doc — MOP claims no predictive distribution/interval/coverage; π\* is a control policy validated on behavior gates (Bench 677), not forecast calibration.
- [ ] **T3.6** **Softmax exemption note** in module docs: π\*'s `exp/Z` normalization is the paper's exact categorical-distribution math (must sum to 1); the house "sigmoid, never softmax" rule governs semantic scalar projections, which this is not. Prevents a future lint/review "fix" from corrupting the math.
- [ ] **T3.7** Verdict + promotion decision: G1–G4 PASS → evaluate default-on (the gain is correctness + capability, modelless by construction); any FAIL → honest bench record `.benchmarks/NNN_mop_primitive_goat.md`, stay opt-in.

## Phase 4 — Handoff

### Tasks

- [ ] **T4.1** Unblock riir-ai Plan 538 (T1.1 there consumes this crate's path dep bump) — cross-reference both plans' status lines.
- [ ] **T4.2** Update Research 478 §3.3 item 1 status: planned → shipped (with bench pointer). One-line PASS-Redirect-style backref not needed (this IS the note's own primitive).

---

## Boundary compliance (pre-checked)

- **Leaf-clean:** no game/chain/shard/emotion vocabulary; inputs are plain arrays + scalars. MOAT gate: katgpt-rs PASS (fundamental inference primitive, Research 478 §3.5).
- **Substrate-first:** consumes `cgsp::types::entropy_nats`; the only prior `MopSolver` (riir-poc) is the defend-wrong PoC reference — retained there as the permanent §3.6 regression check, consumed as parity oracle by riir-ai Plan 538, not duplicated there.
- **Numbers:** feature `mop_path_entropy` allocated per Research 478 §3.3; plan number 573 per `.plans/.highwater` (572 → 573, ls-verified).
