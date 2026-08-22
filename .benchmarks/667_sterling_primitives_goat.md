# Benchmark 667 — Sterling-derived modelless primitives GOAT (Issue 672)

**Date:** 2026-08-19
**Feature:** `sterling_primitives` (opt-in — promotion awaits a consumer GOAT; riir-ai Issue 732 is the first candidate)
**Source:** Research 491 / arXiv:2608.07594 (Steerling-8B, Guide Labs) — the inference-time half.
**Module:** `crates/katgpt-core/src/sterling.rs` + the ungated `noisy_or` / `noisy_or_stable` crate-root utils + `tests/noisy_or_672_util.rs` + `tests/sterling_alloc_check.rs`.

## GOAT gates

| Gate | Evidence | Verdict |
|---|---|---|
| **G1 correctness** | T1 falsifier pair: the NAIVE subtraction promotes anti-aligned tokens (asserted — the paper's Fig. 19 bug reproduced); the gated form leaves `a ≤ 0` tokens **bit-unchanged** (`to_bits` equality). T2 bit-identity: `Σ parts + residual == fused` bit-identical incl. empty-component degenerate cases (0 comps → fused == residual; empty residual → contribution exactly 0; nothing → exactly 0); pre-summed-vector dot agreement < 1e-5 rel at D=2304 (real-arithmetic exactness, honestly fp-quantified — never claimed bit-identical); GEMV variant column-wise bit-identity + scalar-path agreement. T3 lift identities: independence → lift ≈ 1 (≤1e-3); exclusive-to-tagged → > 1, monotone in tagged mass; α→0 zero-tagged-mass → ~0; deterministic top-K (lift DESC / word ASC) bit-stable. T4 noisy-OR boundaries (all-0 → 0 exactly; any-1 → 1 exactly; monotone; **bit-identical to the civ two-term formula over a 64×64 grid** — the delegation's pin); HSIC controls (exactly-orthogonal zero-mean column spaces → gauge exactly 0; identical → max; symmetric; mixed strictly between). | **PASS** |
| **G2 perf** | T1/T2/T3/T4 are O(|V|)/O(n·D)/O(V)/O(M·d²) closed-form loops. Measured (release): the sterling hot-path block (T1 mask + T2 GEMV + T3 bias + T4 gauge + γ-calibration, 512-vocab / D=256 / M=32) runs 10,000 iterations inside the alloc-check in 0.08s total — every op sub-microsecond. | **PASS** |
| **G3 no-regression** | Default lib suite **1897 passed / 0 failed / 6 ignored** (exact pre-existing baseline); `--all-features` check clean; clippy 0 warnings in default + feature-on + `--tests` states; civ salience suite 78/78 with the noisy-OR delegation in place. | **PASS** |
| **G4 alloc-free** | `sterling_alloc_check` (CountingAllocator, release): T1 + T2-GEMV + T3-bias + T4-gauge + γ-calibration × 10,000 iterations = **0 allocations**. (The scalar `decomposed_readout` returns a `Vec` by design — cold path; `LiftTableBuilder` is an offline corpus statistic — cold path.) | **PASS** |

## What shipped (T1–T5)

- **T1 `relu_gated_suppression_into`** — `ℓ ← ℓ − s·ReLU(a)`, branch-free inner loop; plus `tau_over_peak_calibration` (γ = τ/peak(e_c), the logit-space cousin of MAG's `calibrate_alpha`; returns `None` on degenerate directions).
- **T2 `decomposed_readout` + `decomposed_readout_gemv_into`** — exact-decomposition readout for additive consumers; canonical fixed order `(((0+c₀)+c₁)+r)` shared verbatim between scalar and GEMV paths (the invariant is on the returned ledger, by construction bit-identical).
- **T3 `LiftTableBuilder` + `lift_set_to_bias_table`** — two-pass corpus lift statistic → top-K expression targets → log-lift drafter bias table (the consumer demo wiring; the steering-target consumer is `latent_field_steering`'s expression set).
- **T4 riders** — `noisy_or` / `noisy_or_stable` at the crate root (UNGATED — pure math next to `sigmoid`, so the civ delegation adds no feature dep); `hsic_cross_covariance_gauge` (measure-only disentanglement gauge). **Civ site delegates**: `riir-games-civ/src/npc/salience_gate/mod.rs` now calls `katgpt_core::noisy_or(&[c, boost])` — bit-identical (pinned by the 64×64 grid test).
- **T5** — this GOAT + `cargo check --all-features` + clippy clean. **Stays opt-in** per the promotion rule: no consumer GOAT has passed yet (riir-ai Issue 732 — exact-emotion-ledger NPC decisions — is the first candidate; the T2 readout is its substrate).

## Substrate check (substrate-first skill)

Searched: `relu_gated|logit_mask|suppression`, `decomposed_readout|exact_decomposition`, `lift|pmi`, `noisy_or`, `hsic|cross_covariance` across all 7 repos' `*.rs`. Found: unrelated suppression surfaces (write-suppression gates), `cross_covariance` only as a private riir-poc test helper, no lift/PMI statistic, no noisy-OR util (only the civ inline). **Decision: build new** (the issue's grep-verified claim re-confirmed with vocabulary translation). MANCE erasure (activation space) is T1's complement, not duplicate; `soft_bayesian_omega`'s LLR (similarity_inference, private fn) is a different quantity.

## Run

```bash
cargo test -p katgpt-core --features sterling_primitives --lib sterling::       # 9 tests
cargo test -p katgpt-core --test noisy_or_672_util                              # 5 tests (default build)
cargo test -p katgpt-core --features sterling_primitives --test sterling_alloc_check --release
```

## Honest caveats

- The paper's trained-absence / steering-responder halves are NOT distilled (genuinely trained — Research 491 routes them to riir-train Research 425).
- `lift_set_to_bias_table` applies log-lift biases; consumers must pin their own gain (corpus-scale-dependent).
- Promotion deliberately NOT requested — awaiting the riir-ai Issue 732 consumer GOAT.
