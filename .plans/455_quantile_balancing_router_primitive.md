# Plan 455: Quantile Balancing MoE Router (Modelless)

**Date:** 2026-07-17
**Research:** [katgpt-rs/.research/447_Kimi_K3_KDA_AttnRes_LatentMoE.md](../.research/447_Kimi_K3_KDA_AttnRes_LatentMoE.md) §2.4
**Source algorithm:** [Jianlin Su, Feb 2026 blog](https://spaces.ac.cn/archives/11619) + [Marin team JAX validation at 32B-A5B / 1e22 FLOPs](https://openathena.ai/blog/quantile-balancing/)
**Target:** `katgpt-rs/crates/katgpt-spectral/src/quantile_balance_router.rs` (new module, sibling to `manifold_power_iter_router.rs`) + Cargo feature `quantile_balance_router` (re-exported from root `katgpt-rs/Cargo.toml` as `quantile_balance_router = ["katgpt-spectral/quantile_balance_router"]`).
**Status:** Skeleton (not yet started).

> **Numbering note.** The research note (447 §6) proposed this as `Plan 447`. That number is already in use (`.plans/447_freq_bandit_phase1.md`), and the katgpt-rs AGENTS.md rule — *"monotonic and never reused — even after a file is removed per the noise-reduction rule"* — forbids recycling it. Re-issued as **Plan 455** = `.plans/.highwater` (454) + 1. The matching KDA binding issue was re-issued as **Issue 179** (not 165 — `.issues/165_dd_tree_file_split_c2.md` exists).

---

## Goal

Distill Research 447 §2.4 into a generic, modelless, MIT-licensed Rust module that performs **one-shot MoE router-bias balancing at freeze/thaw snapshot swap** (mirrors Plan 279's application point). Concretely: given a frozen router score matrix `s ∈ ℝ^{m×n}` (m tokens, n experts), compute a per-expert bias vector `β ∈ ℝⁿ` via alternating-coordinate descent on the balanced-assignment LP, then route as `top-k(s − β)`. Zero hyperparameters (paper §2), validated at Marin 32B-A5B / 1e22 FLOPs.

**Sibling to Plan 279 Manifold Power Iteration Router**, not a replacement. The two answer different questions on the same snapshot-swap boundary:

- **Plan 279 (MPI)** — *"this expert pool is poorly aligned, fix the router **rows** once."* Operates on `R ∈ ℝ^{N×D}` against per-expert Gram matrices. Improves router–expert **alignment** (λ).
- **Plan 455 (QB)** — *"this expert pool is well-aligned but load is unbalanced, fix the **bias** per snapshot."* Operates on `s ∈ ℝ^{m×n}` for a calibration batch drawn from the new pool. Improves load-**balance** (MaxVio).

Phase 3 of this plan runs both on the same synthetic pool and promotes whichever wins the joint λ/MaxVio comparison to default-on.

**Inference-only reframing (honest caveat).** QB is published as a **per-step training** algorithm (bias updated every step from the current batch's router scores). We don't train. The distillation reframes QB as a **snapshot-swap one-shot bias computation**: when the expert pool changes, we run QB once on a calibration batch (a fixed, frozen set of representative router-score rows committed with the snapshot), compute `β`, and ship `β` alongside the snapshot. The LP formulation is application-agnostic so the math transfers faithfully; **but Marin's 1e22-FLOPs empirical validation was for the per-step variant, not the snapshot-swap variant** (research note §5 caveat 6). The GOAT gate G8 (revalidation at the new application point) is therefore non-negotiable — see Phase 2.

---

## Phase 1 — Unblocking Skeleton (CORE — required to proceed with anything else)

Goal: a compiling, tested, feature-gated module that implements `quantile_balance_router` on synthetic data with the public API surface frozen. No snapshot-swap wiring yet (Phase 2 of this plan does NOT need a snapshot hook — see §"Why QB needs no snapshot hook" below; it is a pure function from `s` to `β`).

### Tasks

- [ ] **T1.1** Add feature flag `quantile_balance_router = []` to `katgpt-rs/crates/katgpt-spectral/Cargo.toml` features section (alphabetical, near `manifold_power_iter_router`). Add a root-level alias `quantile_balance_router = ["katgpt-spectral/quantile_balance_router"]` to `katgpt-rs/Cargo.toml` (mirror Plan 279's root-facade pattern).
- [ ] **T1.2** Add `#[cfg(feature = "quantile_balance_router")] pub mod quantile_balance_router;` to `katgpt-rs/crates/katgpt-spectral/src/lib.rs` (alphabetical, near `manifold_power_iter_router`).
- [ ] **T1.3** Implement `QbConfig` struct in `quantile_balance_router.rs`:
  - `iters: u8` (=5 default per Su blog reference NumPy impl; the LP converges in 1–5 steps)
  - `causality_strict: bool` (=true: use old `β` to select experts for the calibration batch, THEN update `β`. False = leak-future-info trap per Su blog §"小心陷阱")
- [ ] **T1.4** Implement `QbResult` struct: `beta: Vec<f32>` (n), `alpha: Vec<f32>` (m, diagnostic only — per-token Lagrange multiplier, discarded at inference), `final_balance_violation: f32` (MaxVio diagnostic), `converged_iter: u8` (early-stop detection).
- [ ] **T1.5** Implement `pub fn quantile_balance_router(s: &[f32], m: usize, n: usize, k: usize, cfg: &QbConfig, scratch: &mut QbScratch) -> QbResult`:
  - **Algorithm** (Su blog + Marin JAX, verbatim translation):
    ```
    beta ← zeros(n)
    for t in 0..cfg.iters:
        # per-token Lagrange multiplier (per-row quantile of de-biased scores)
        for i in 0..m:
            row = s[i*n..(i+1)*n] - beta
            alpha[i] = quantile(row, 1 - k/n)         # the (1 - k/n) quantile
        # per-expert dual (per-column quantile of de-biased scores)
        for j in 0..n:
            col[j] = s[j, *] - alpha[*]               # materialize column
            beta[j] = quantile(col[j], 1 - k/n)
        if ||beta_new - beta_old||_∞ < tol: break     # early-stop
    return beta
    ```
  - **Quantile computation:** use the `pdqselect`-style `nth_element` already in `katgpt-core` if available; else ship a small branchless `partition_select` (no `sort` — quantile needs `O(n)` not `O(n log n)`). Grep `katgpt-core/src/` for existing quantile/partition helpers before reinventing.
  - **Zero-alloc on hot path:** `QbScratch { row_buf: Vec<f32>, col_buf: Vec<f32>, beta_prev: Vec<f32> }` — caller-owned, reused across iterations. Document allocation pattern in module header.
- [ ] **T1.6** Implement `pub fn route_with_bias(s_row: &[f32], beta: &[f32], k: usize, out_scores: &mut [f32]) -> Vec<usize>` — apply bias then top-k. **Sigmoid discipline (AGENTS.md):** the output gate (if used by a downstream consumer that wants sigmoid scoring) is `σ(s_row[j] - beta[j])` per-expert independent, then TopK_k. Never softmax. The bias itself is just a subtraction — no activation. Mirror Plan 279 `gate_sigmoid_topk` exactly so consumers can swap MPI-conditioned rows ↔ QB-biased scores at the routing layer.
- [ ] **T1.7** Write unit tests in `quantile_balance_router.rs` `mod tests`:
  - [ ] **G1 mechanics:** output `β` shape matches `n`; output is deterministic given `(s, m, n, k, cfg)`
  - [ ] **G2 perfect-balance check:** on a synthetic `s` where vanilla top-k produces `MaxVio > 0`, verify `MaxVio(s − β) ≤ 0.1·MaxVio(s)` (the LP solution drives MaxVio → 0; gate at 0.1 to absorb quantile-rounding noise on integer-count constraints)
  - [ ] **G3 zero-degradation on already-balanced input:** on `s` where vanilla top-k is already balanced, verify `MaxVio(s − β) ≤ MaxVio(s)` (QB never makes balance worse — LP optimum is at worst the no-op `β = 0`)
  - [ ] **G4 causality trap:** with `causality_strict=true`, verify the calibration-batch expert selection uses `β_old`, not `β_new`. (Test by injecting an adversarial `s` where using `β_new` would create a circular update.)
  - [ ] **G5 convergence:** verify `iters=5` and `iters=10` produce `β` within `1e-4` relative error on a synthetic instance (the LP converges in 1–5 steps per Su blog).
  - [ ] **G6 zero-row safety:** degenerate `s` (all-zero row) → `β = 0`, no panic (mirror Plan 279 `test_*_zero_matrix_safe`).
- [ ] **T1.8** Add example `examples/quantile_balance_router_basic.rs`:
  - [ ] Synthetic MoE: N=8 experts, M=64 calibration tokens, k=2
  - [ ] Construct `s` with deliberately skewed expert affinity (expert 0 picked 32/64 times, expert 7 picked 0/64 times → MaxVio high)
  - [ ] Compute `β`, print `MaxVio(s) → MaxVio(s − β)` (target: large → ≈0)
  - [ ] Print timing (target: sub-ms for N=8, M=64)
  - [ ] Show `route_with_bias` on a sample token
- [ ] **T1.9** Document module in `quantile_balance_router.rs` header with:
  - [ ] Paper/blog reference (Su blog + Marin JAX validation link)
  - [ ] Algorithm (LP formulation + minimax + alternating-coordinate descent, copied from Research 447 §2.4)
  - [ ] Causality trap warning (Su blog §"小心陷阱")
  - [ ] Sibling-relationship note with Plan 279 (rows vs bias; alignment vs balance)
  - [ ] Inference-only reframing caveat (per-step training → snapshot-swap one-shot)

### Phase 1 Exit Criteria
- [ ] `cargo build --features quantile_balance_router -p katgpt-spectral` compiles clean
- [ ] `cargo build --features quantile_balance_router` (root workspace) compiles clean
- [ ] `cargo test --features quantile_balance_router -p katgpt-spectral --lib quantile_balance_router` passes all unit tests
- [ ] `cargo run --example quantile_balance_router_basic --features quantile_balance_router --release` runs and prints MaxVio before→after
- [ ] `cargo clippy --features quantile_balance_router --all-targets -p katgpt-spectral` zero new warnings
- [ ] File sizes < 2048 lines (target: `quantile_balance_router.rs` < 600)

---

## Phase 2 — GOAT Gate Benchmark

Goal: prove the QB algorithm's claims on a real calibration-batch shape before any promotion decision. Per AGENTS.md: every new primitive ships behind a feature flag + benchmark proving the gain.

### Tasks

- [ ] **T2.1** Create `benches/quantile_balance_router_bench.rs` in `katgpt-spectral/benches/` (std::time::Instant, not criterion — matches Plan 279 `manifold_power_iter_router_bench.rs` style):
  - [ ] Sweep `N ∈ {8, 32, 64, 256}` experts, `M ∈ {64, 256, 1024}` calibration tokens, `k ∈ {1, 2, 4}`
  - [ ] Measure: per-iter cost, total `β` compute time, `route_with_bias` per-token cost
  - [ ] Print `MaxVio(s) → MaxVio(s − β)` for each `(N, M, k)`
- [ ] **T2.2** Create `tests/bench_455_quantile_balance_goat.rs` — the GOAT gate test file:
  - [ ] **G1 — Mechanics:** `β` shape matches `n`, deterministic given inputs, no NaN/Inf in `β` for any well-formed `s`
  - [ ] **G2 — MaxVio reduction:** `MaxVio(s − β) ≤ 0.1·MaxVio(s)` on a deliberately-skewed synthetic `s` (LP drives MaxVio → 0; 0.1 absorbs quantile-rounding noise)
  - [ ] **G3 — No-degradation on balanced input:** `MaxVio(s − β) ≤ MaxVio(s)` on already-balanced `s`
  - [ ] **G4 — Sub-ms swap cost at game scale:** `N=8, M=256, k=2` (typical NPC LoRA pool + calibration batch) total `β` compute time < 1ms on commodity CPU (release build)
  - [ ] **G5 — Determinism / sync-safety:** same `(s, m, n, k, cfg)` → byte-identical `β` across two independent runs (quorum-safe, sync-block-safe)
  - [ ] **G6 — Sigmoid constraint (AGENTS.md):** output gate uses independent per-expert sigmoid (if used), never softmax. Static check + runtime assertion that changing one expert's score does not perturb others (mirror Plan 279 G7).
  - [ ] **G7 — `iters=5` sufficiency:** verify `iters=5` and `iters=10` produce `β` within `1e-4` relative error (LP converges in 1–5 steps). Gate `iters=5` as default; demote `iters>5` paths.
  - [ ] **G8 — Snapshot-swap revalidation (the honest caveat):** construct a calibration batch from a frozen snapshot's representative tokens (NOT a live training step's tokens). Verify `MaxVio` reduction holds at the snapshot-swap application point — i.e., that the per-step-validated algorithm still works when the bias is computed once on a fixed batch and reused for many inference tokens. **This gate exists because Marin's 1e22-FLOPs validation was for the per-step variant, not the snapshot-swap variant (Research 447 §5 caveat 6).** If G8 fails: keep QB opt-in, document that snapshot-swap application requires either (a) a larger calibration batch, (b) periodic re-biasing, or (c) a hybrid with MPI.
- [ ] **T2.3** Add GOAT gate summary print at end of `bench_455_*_goat.rs`: count G1–G8 pass/fail, exit code non-zero if any fail.

### Phase 2 Exit Criteria
- [ ] G1–G8 all green on release build
- [ ] GOAT gate summary table in this plan's "GOAT Gate" section below is filled in with measured numbers
- [ ] G8 explicitly addresses the snapshot-swap revalidation caveat (the honest part of this plan)

---

## Phase 3 — Head-to-Head vs Plan 279 MPI Router + Promotion

Goal: per AGENTS.md promotion rule — run both QB (this plan) and MPI (Plan 279) on the same synthetic expert pool, pick the winner on the joint `(λ, MaxVio)` Pareto frontier, promote to default, demote the loser. If neither Pareto-dominates, ship both as siblings and let the consumer pick via feature flag.

### Tasks

- [ ] **T3.1** Construct shared synthetic MoE test fixture: `N=8 experts, D=256, M=256 calibration tokens`. The fixture must have **both** misalignment (low λ) **and** imbalance (high MaxVio) — a deliberately hard case where both algorithms should help.
- [ ] **T3.2** Run **Plan 279 MPI** on the fixture: produces `R' ∈ ℝ^{N×D}` (router rows reconditioned). Measure `λ(R')` and `MaxVio(R')`.
- [ ] **T3.3** Run **Plan 455 QB** on the fixture: produces `β ∈ ℝⁿ` (per-expert bias). Measure `λ(s − β)` and `MaxVio(s − β)`.
- [ ] **T3.4** Run **both composed** (MPI then QB): `R' → s' → β' → MaxVio(s' − β')`. Research 447 §2.4 last bullet says these are *complementary not competing* — verify by measuring whether composition strictly beats either alone.
- [ ] **T3.5** Decision matrix (fill in after T3.2–T3.4):

  | Variant | λ | MaxVio | Verdict |
  |---|---|---|---|
  | Vanilla (no conditioning) | TBD | TBD | baseline |
  | MPI only (Plan 279) | TBD | TBD | improves λ |
  | QB only (Plan 455) | TBD | TBD | improves MaxVio |
  | MPI + QB (composed) | TBD | TBD | if strictly Pareto-better, promote both |

- [ ] **T3.6** **Promotion decision:**
  - [ ] **Case A — QB Pareto-dominates MPI** (better MaxVio, equal-or-better λ): promote `quantile_balance_router` to DEFAULT-ON, demote `manifold_power_iter_router` to opt-in. Update both plans + research notes.
  - [ ] **Case B — MPI Pareto-dominates QB**: keep QB opt-in, document the regime where QB wins (research note §2.4 predicts QB wins on skewed distributions where MPI's row-retraction can't fix imbalance). No demotion of MPI.
  - [ ] **Case C — Composition strictly beats either alone** (the predicted outcome per Research 447 §2.4): promote **both** to DEFAULT-ON. Document that they solve orthogonal problems (alignment vs balance) and ship them as a composed pipeline: `R' = MPI(R, M)` then `β = QB(s_with_R', calibration_batch)`. Add a `composed_router_pipeline` example.
  - [ ] **Case D — Tie / inconclusive**: ship both as opt-in siblings. Consumer picks via feature flag. Document the regime boundary (e.g., "QB for skewed expert-affinity distributions; MPI for poorly-aligned router-row geometry").
- [ ] **T3.7** Update research note `katgpt-rs/.research/447_*.md` Status field: `Active → Done` (if promoted) or `Active → Shelved` (if demoted). Add a one-line postscript: "Plan 455 GOAT gate: N/8 green + head-to-head outcome X, promoted|shelved|sibling on YYYY-MM-DD."
- [ ] **T3.8** Update `README.md` Feature Showcase + GOAT Proofs section with the head-to-head outcome.

### Phase 3 Exit Criteria
- [ ] Head-to-head decision matrix filled in
- [ ] Promotion decision recorded in this plan + research note + README
- [ ] If Case A or C: default feature set updated in `Cargo.toml`
- [ ] If Case C: composed-pipeline example shipped

---

## GOAT Gate (pass criteria)

| Gate | Metric | Target | Our threshold | Status |
|------|--------|--------|---------------|--------|
| **G1** | Mechanics (β shape, determinism, finiteness) | n/a | shape + no NaN/Inf + byte-identical | ⏳ |
| **G2** | MaxVio reduction on skewed input | → 0 | `MaxVio(s − β) ≤ 0.1·MaxVio(s)` | ⏳ |
| **G3** | No-degradation on balanced input | `≤ MaxVio(s)` | `MaxVio(s − β) ≤ MaxVio(s)` | ⏳ |
| **G4** | Swap cost at game scale | sub-ms | `N=8, M=256, k=2` total < 1ms release | ⏳ |
| **G5** | Determinism / sync-safety | byte-identical | same inputs → same `β` across runs | ⏳ |
| **G6** | Sigmoid constraint (AGENTS.md) | sigmoid gate, never softmax | static + runtime check | ⏳ |
| **G7** | `iters=5` sufficiency | LP converges 1–5 steps | `‖β_5 − β_10‖/‖β_5‖ < 1e-4` | ⏳ |
| **G8** | Snapshot-swap revalidation (the honest caveat) | per-step algo works at snapshot point | `MaxVio` reduction holds on frozen calibration batch | ⏳ |

**Promotion rule (AGENTS.md):** G1–G8 all green + Phase 3 head-to-head outcome is Case A or C → promote. Any red OR Phase 3 Case B/D → keep opt-in, document the regime boundary. **G8 is non-negotiable** — the inference-only reframing is the honest part of this plan; if per-step training validation doesn't transfer to snapshot-swap, we say so.

---

## DRY Note

`quantile_balance_router` (this plan) and `manifold_power_iter_router` (Plan 279) are both **one-shot deterministic MoE router reconditioners applied at freeze/thaw snapshot swap**. They are NOT instances of the same operation (no shared helper needed for Phase 1):

- **MPI** — retracts router **rows** against expert Gram matrices via power iteration. Helper: `power_iter_retract` (already shared with `gauge_rebalance` Plan 270).
- **QB** — solves the balanced-assignment LP via alternating-coordinate descent on per-row/per-column quantiles. Helper: a small `partition_select` (nth-element) — **grep `katgpt-core/src/` for existing quantile/partition helpers before reinventing**.

If Phase 3 Case C (composition wins) lands, the natural DRY abstraction is a `RouterReconditioner` trait with two impls:

```rust
pub trait RouterReconditioner {
    fn recondition(&mut self, ctx: &RouterCtx) -> RouterResult;
}
```

where `RouterCtx` carries `(R, expert_grams, calibration_batch)` and each impl picks what it needs. **Do NOT extract this trait in Phase 1** — premature abstraction before we have empirical evidence (Phase 3) that the two algorithms are composed in practice.

---

## Why QB needs no snapshot hook (unlike Plan 279 Phase 2)

Plan 279 ships a `MpiRouterSnapshotHook` trait + default impl because the Gram matrices `M[i]` must be cached + invalidated on snapshot version bump. QB has **no analogous per-snapshot state**: `β` is a pure function of `(s, k, cfg)` where `s` is the calibration-batch router scores. The caller (riir-ai's freeze/thaw runtime) supplies `s` at the snapshot-swap boundary; the QB module is just `fn(s) → β`. Adding a hook trait here would duplicate Plan 279's surface without adding capability. **Skip Phase 2 entirely** — Phase 1's pure function IS the snapshot-swap API. riir-ai wires it via a one-liner call at the swap boundary, exactly like Plan 279 T2.1's hook does for MPI.

If empirical evidence later shows we want cached `β` across swap-then-inference cycles (e.g., EMA over multiple snapshots to smooth bias drift), that's a follow-up issue — out of scope here.

---

## Out of Scope (Deferred / riir-train / Future)

- **Per-step training variant of QB** (Su blog's original formulation, bias updated every step from the live batch) → `riir-train`. The per-step variant is what Marin validated at 1e22 FLOPs; the snapshot-swap variant is our distillation. One line: **QB per-step training → riir-train**.
- **Quantile selection algorithm micro-optimization** (e.g., AVX-512 partition, branchless Hoare). Phase 1 ships a simple `partition_select`; SIMD quantile can be a follow-up bench optimization if G4 fails.
- **Composed `RouterReconditioner` trait** — see DRY Note above. Premature until Phase 3 evidence.
- **Live rebalancing** (recompute `β` every N tokens during inference based on observed router scores) — violates freeze/thaw constraint. Out of scope.
- **Cross-validation across snapshot distributions** (does `β` from snapshot A transfer to a slightly-different snapshot B without recompute?) — speculative robustness question, follow-up issue if Case C lands.
- **Numerical stability under degenerate `s`** (all rows identical, all experts identical, `k=n`) — covered by T1.7 G6 zero-row safety; deeper edge-case fuzzing is a follow-up.

---

## File Layout (target)

```
katgpt-rs/
├── Cargo.toml                                            # +alias quantile_balance_router = ["katgpt-spectral/quantile_balance_router"]
└── crates/katgpt-spectral/
    ├── Cargo.toml                                        # +feature quantile_balance_router = []
    ├── src/
    │   ├── lib.rs                                        # +mod quantile_balance_router
    │   └── quantile_balance_router.rs                    # NEW — QB primitive + sigmoid-biased gate (pure fn)
    ├── examples/
    │   └── quantile_balance_router_basic.rs              # NEW — MaxVio before/after demo + timing
    └── tests/
        └── bench_455_quantile_balance_goat.rs            # NEW — GOAT gate G1–G8
```

Sibling to (NOT inside) `crates/katgpt-spectral/src/manifold_power_iter_router.rs`. The two files live side-by-side; Phase 3 composes them via the caller (or a `RouterReconditioner` trait if Case C lands — TBD).

---

## Constraints Checklist

- [ ] **Modelless first** — one-shot `β` computation at snapshot swap. No backprop, no weight mutation during inference, no per-token `β` update.
- [ ] **Latent-to-latent with sigmoid** — `route_with_bias` uses independent per-expert sigmoid (G6) if a downstream consumer wants sigmoid scoring. The bias itself is a subtraction, no activation. Never softmax.
- [ ] **Freeze/thaw** — `β` is computed at the snapshot-swap boundary from a frozen calibration batch. Never mutated during inference.
- [ ] **File < 2048 lines** — target `quantile_balance_router.rs` < 600.
- [ ] **SOLID / zero-alloc hot paths** — caller-owned `QbScratch`, no allocation in the alternating-coordinate loop.
- [ ] **CPU/SIMD** — plasma/hot tiers via existing `partition_select`. SIMD quantile is a follow-up if G4 fails.
- [ ] **Determinism / sync-safety** — same `(s, m, n, k, cfg)` → byte-identical `β`. Safe under `SyncBlock → ChainConsensus` quorum (G5).
- [ ] **4-repo discipline** — engine primitive in katgpt-rs (MIT, no game IP); runtime wiring (snapshot-swap hook) in riir-ai; chain transport in riir-chain; per-step training variant in riir-train.
- [ ] **GOAT gate** — G1–G8 defined; promote to default iff 8/8 green AND Phase 3 head-to-head is Case A or C.
- [ ] **`Uuid::now_v7()` / blake3 / argon2 / papaya** — N/A for this primitive. (If a future Phase adds snapshot-version-keyed `β` cache, use BLAKE3 of the calibration batch — mirror Plan 279 T2.3.)
- [ ] **Honest caveats** — G8 (snapshot-swap revalidation) is non-negotiable. The per-step → snapshot-swap reframing is the only place this plan departs from the paper's empirical validation, and it gets a dedicated GOAT gate.

---

## TL;DR

Plan 455 ships a modelless, MIT-licensed `quantile_balance_router` primitive that computes a per-expert bias vector `β ∈ ℝⁿ` via alternating-coordinate descent on the balanced-assignment LP (Su blog Feb 2026 + Marin 32B-A5B / 1e22-FLOPs validation), applied **once per freeze/thaw snapshot swap** as a sibling to Plan 279 Manifold Power Iteration Router. Three phases: (1) unblocking skeleton — pure `fn(s, k, cfg) → β` + unit tests, no snapshot hook needed (the pure fn IS the snapshot API); (2) GOAT gate G1–G8 with **G8 = snapshot-swap revalidation** as the non-negotiable honest check (Marin validated per-step, we apply snapshot-swap — the math transfers but the empirical claim doesn't, so G8 must re-prove it); (3) head-to-head vs Plan 279 MPI on a deliberately-hard synthetic pool — predicted outcome per Research 447 §2.4 is Case C (composition wins, both promoted to default as orthogonal axes: MPI fixes alignment λ, QB fixes balance MaxVio). Zero hyperparameters (Su blog §2). Sigmoid discipline preserved on the gate output.
