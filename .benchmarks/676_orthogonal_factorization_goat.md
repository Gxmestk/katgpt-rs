# Bench 676 — Orthogonal Factorization Primitives GOAT (Issue 687)

> **Primitive:** `katgpt-core` `orthogonal_factorization` (opt-in, implies
> `spectral_pencil`) — `orthonormalize_into` + `orthogonality_defect` (T1),
> `FactorActivityScratch` + `factor_activity_hinge` + `gamma_schedule` (T2),
> `parseval_energy_check` + `recompose_into` + `kept_energy` +
> `hadamard_factorize` (T3), `head_conditioning` + `rollout_bound` (T4).
> **Source:** Research 504 (arXiv:2608.20065 "Orthogonal JEPA") Path 0 — the
> paper's *structure* (orthonormal-by-construction bases, activity hinges,
> Parseval invariants, conditioning certificates) with no gradient anywhere.
> **Plan:** `.plans/579_orthogonal_factorization_primitives.md`.
> **Date:** 2026-08-25. **Box:** M3 Max (aarch64), release, load avg 4.3–6.6
> (sibling riir-train build concurrent — G2 measured UNDER contention, the
> conservative direction).
> **Run:** `cargo test --release -p katgpt-core --features
> orthogonal_factorization --bench bench_687_orthogonal_factorization_goat --
> --nocapture`

## Verdict table

| Gate | Target | Measured | Verdict |
|---|---|---|---|
| **G1** determinism ×3 | bit-identical GS output/defect/hinge/parseval bits | PASS (all three runs identical) | **PASS** |
| **G1** Hadamard dyadic anchors | Parseval residual EXACTLY 0.0; recompose bit-exact == z; basis defect EXACTLY 0.0 (d=64) | `true \| true \| true` (residual printed `0e0`) | **PASS** |
| **G2** GS latency | < 5,000 ns/call @ d=64/K=14 | **4,881 ns/call** (under load) | **PASS** (2.4% margin) |
| **G2** hinge observe latency | amortized O(N·d), gate < 5,000 ns/sample | **21 ns/sample** @ K=8×r=8 (Kr=64, 1000-sample window); 8 ns/sample @ K=14×r=1 (drive shape) | **PASS** |
| **G3** no-regression | default + feature-on + standalone-chain clean | default lib **1913/0/7i**; `+orthogonal_factorization` lib **1989/0/8i** (24 new module tests); `--no-default-features --features orthogonal_factorization` compiles clean; default clippy `-D warnings` clean | **PASS** |
| **G4** zero steady-state alloc | 0 allocs / 100 mixed calls | **0** (module lib test `g4_zero_alloc_steady_state`, TrackingAllocator — GS + observe + hinge + parseval + recompose + kept_energy + hadamard + head_conditioning) | **PASS** |
| **G8a** planted near-parallel pair | input defect > 0.1 AND > 100× healthy; GS output \|cos\| < 1e-6; survivor unit-norm | healthy defect **5.2e-15**, planted **0.9957** (ratio **1.9e14×**); max \|cos\| = **1.38e-8**; survivor ‖b‖² = 1.000000 | **PASS** |
| **G8b** planted dead channel | hinge fires EXACTLY on (k=3, j=5) with value == γ (bit-exact); healthy coords exactly 0 | per[29] bits == γ bits (**true**), worst_flat == **29**, mean = 0.003906 = γ/64 | **PASS** |

**ALL GATES PASS.** Informational: `head_conditioning` 8 heads @ d=64 =
5.8 ms total (construction-time by design — exact pinned Jacobi per head;
not a hot-path gate). `rollout_bound` exact (`powi`).

## Unit-test record (24 module tests, all green)

`gs_orthonormalizes_random_set`, `defect_closed_form_tiny` (hand-computed
L_orth: non-unit (4−1)²=9, parallel cross=1, short 0.5625 — all EXACT),
`empty_set_noop`, `gs_zeroes_exact_duplicate`, `gs_zeroes_overflow_beyond_rank`
(d=4/K=6 ⇒ 2 rank-spill rows zeroed), `gs_preserves_span_complete_basis`,
`hadamard_defect_zero_at_d64_tiny_at_d8` (d=64 EXACTLY 0.0 — the dyadic-scale
point; d=8 rounds at 1/√8), `parseval_exact_and_recompose_bit_identical_at_d64`,
`parseval_catches_duplicate_and_incomplete`, `truncation_certificate_identity`
(‖z − recompose(kept)‖² == total − kept), `defect_fires_on_planted_near_parallel_pair`,
`gamma_schedule_values` (exact: max(0.25, 1/√n) at n ∈ {0,1,4,16,10⁶}),
`hinge_matches_two_pass_variance` (vs direct two-pass f64, both hinge arms),
`hinge_dead_channel_exactly_gamma` (bit-exact), `hinge_no_claim_below_two_samples`,
`head_norm_diag_exact` (diag(3,4,½) → 4.0), `head_norm_rank_one` (‖u‖‖v‖),
`head_norm_orthonormal_is_exactly_one` (Hadamard d=64 Gram == I exactly),
`conditioning_cert_and_rollout_bound` (σ_max=9 worst head 2; 2¹⁰=1024),
`g4_zero_alloc_steady_state`, + 4 `should_panic` shape guards.

## Implementation notes (the honest deltas from the issue sketch)

1. **Feature skeleton refinement (documented deviation).** The issue sketched
   `orthogonal_factorization = []`; T4's certificate is specified *via
   `spectral_pencil`* — consuming the pinned Jacobi (substrate-first, no
   duplicated eigensolver) requires `orthogonal_factorization =
   ["spectral_pencil"]`. Verified standalone:
   `--no-default-features --features orthogonal_factorization` compiles.
2. **`defect` = the INPUT set's L_orth** (not the output's). G8a's "defect
   fires" is a statement about the audited direction set; the OUTPUT basis's
   own defect is ≤ ~1e-12 by construction (pinned by unit test). A standalone
   `orthogonality_defect(vectors)` audits any production set without
   orthogonalizing.
3. **Precision split (GS core).** Dots/norms accumulate in **f64** with a
   source-fixed 8-lane pattern (element i → lane i%8, lanes summed in
   ascending order — bit-identical across runs AND platforms while restoring
   add-latency ILP; Rust forbids LLVM reassociation, so the lane pattern is
   written in source). The elementwise projection update runs in **f32** with
   the f64-derived coefficient — house precedent (every shipped GS fixture in
   this repo is pure f32) — and the second reorthogonalization pass cleans
   the extra rounding: measured max |cos| = 1.4e-8 vs the 1e-6 gate. The
   all-f64-elementwise variant measured 13.0 µs → 5.4 µs (scalar f64
   round-trips dominated); the split lands 4.88 µs. Dyadic anchors are exact
   under ANY association (no rounding at any width) — unaffected by
   construction.
4. **Exactness witness.** d=64 Hadamard (scale 0.125 = 2⁻⁶, dyadic) with
   dyadic z (multiples of 0.25, |z| ≤ 4): every intermediate carries ≤ 22
   significand bits — Parseval residual EXACTLY 0.0, recompose bit-exact,
   Gram == I exactly (head norm exactly 1.0). d=8 (odd exponent, 1/√8
   irrational) is correctly-rounded but not exact — documented at
   `hadamard_factorize`.
5. **γ schedule constants** (`GAMMA_FAC_MIN` = 0.25, `GAMMA_SCHED_C` = 1.0):
   c = 1.0 keeps γ ≥ 1/√n a factor √2 above the Gaussian σ̂ sampling noise
   σ/√(2n) at unit σ for every n; the floor 0.25 binds from n = 16 on. Both
   overridable per call.

## Promotion decision

**Stays opt-in** (the no-default-consumer rule). This is the modelless half
only; the GOAT here is primitive-level. Consumer wiring is deliberately
out of scope per the issue's Non-goals: riir-ai affect-direction
orthogonalization A/B is a **gameplay owner call** (the CLR precedent —
Bench 010/011 same-day promote-then-demote), riir-neuron-db blend
interference gate / ShardCompactor merge criterion, and riir-train Plan 351
(the trained half, whose Phase-2 hinge A/B docks onto the Bench 494
dist-guard harness) file their own issues against THIS gate's passing
record.

## Resolution

Issue 687 **RESOLVED** — all six tasks (T1–T6) complete; this bench is the
T5/T6 record. Commits: see `feat:` (module + wiring + GOAT bench) and
`docs:` (this record + plan 579 + issue removal) on `develop`. Issue file
removed per the noise-reduction rule; the permanent record is this doc +
Plan 579 + the Cargo.toml feature comment.
