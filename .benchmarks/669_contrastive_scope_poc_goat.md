# Benchmark 669 — Contrastive scope-gate POC GOAT (Issue 674)

**Date:** 2026-08-19
**Feature:** `contrastive_scope` (opt-in POC — **T5 verdict: shipped opt-in, promotion deferred pending consumer adoption** per the issue's own rule)
**Source:** Research 493 / arXiv:2608.13545 (LittleLearner, Li et al., MPI-IS/ETHZ).
**Module:** `crates/katgpt-core/src/contrastive_scope.rs` + `tests/contrastive_scope_alloc_check.rs`.

## GOAT gates

| Gate | Evidence | Verdict |
|---|---|---|
| **G1 parity** | Toy-corpus parity vs hand-computed smoothed log2 ratios (both in- and out-characteristic words, ≤1e-6); zero-count edge: a word in NEITHER corpus scores **exactly 0** (neutral word); α→0 limit sharpens contrast monotonically with correct signs and no NaN; commitment determinism (identical builds ⇒ identical BLAKE3) + tamper-sensitivity (one changed doc ⇒ different commitment); freeze/thaw bit-round-trip incl. malformed-input rejection. | **PASS** |
| **G2 perf** | `t2_scope_score` (release): **6.893 µs/doc at 10⁴ tokens** (document-order fixed accumulation — the bit-identity order is the API contract, so no SIMD re-association). | **PASS** |
| **G3 no-regression (the load-bearing half)** | **In-distribution bit-identity**: for in-scope docs with κ·|D| ≥ 40 the f32 `fast_sigmoid` early-exits to exactly 1.0 ⇒ `haircut == c` **bit-identical** (asserted with `to_bits`); OOS fixtures discounted to exactly 0 (mirror saturation) and declined at θ; mixed docs strictly discounted; decline semantics strict (`D == θ` NOT declined). Default lib suite 1897/0/6 exact; `--all-features` clean; clippy 0. | **PASS** |
| **G4 alloc-free** | `contrastive_scope_alloc_check` (CountingAllocator, release): `scope_score` + `scope_score_from_pairs` + gate × 10,000 = **0 allocations**; the OOS probe battery's only per-call allocation is its documented report `Vec` (≤2, asserted). | **PASS** |
| **T4 battery** | Clean battery: `mean_d_in < 0`, `mean_d_out > 0`, in-side haircut unchanged, out-side decline-rate ≥ 0.9, zero leak suspects. **Seeded leak caught**: one out-scope doc injected at in-probe index 3 ⇒ `leak_suspects == [3]` exactly. | **PASS** |

## What shipped (T1–T4)

- **T1 `ContrastiveScoreBuilder` → `ContrastiveScoreTable`** — two-pass streaming counts (in/out corpora) → smoothed log2-odds `score(w) = log2((c_B+α)/(N_B+αV)) − log2((c_I+α)/(N_I+αV))`; BLAKE3-committed over the canonical builder serialization; freeze/thaw round-trip. **Design deviation (documented in-module): no papaya dep** — the built table is immutable (lock-free reads by construction, `Arc`-shareable); papaya is reserved for a live-update consumer integration, not the POC.
- **T2 `scope_score` / `scope_score_from_pairs`** — `D(x) = Σ tf·score` (Naive-Bayes log-LLR sparse GEMV), document-order fixed accumulation (bit-identical to the scalar loop by construction).
- **T3 `ScopeGate` / `ScopeVerdict`** — epistemic haircut `ĉ = c·sigmoid(−κ·D)` + decline wiring (`D > θ`). κ/θ defaults are POC-scale; consumers re-pin per their benches (the issue's honest caveat).
- **T4 `oos_probe_battery`** — the paired in/out probe report (mean D per side, haircuts, decline rate, seeded-leak suspects) — the Report-the-Floor `cov_in − cov_out` axis extension.

## T5 verdict (the issue's own rule)

> "if G-gates pass AND a consumer adopts (riir-clippy L4 2D gate or riir-ai engram), promote per GOAT discipline; else record negative result and close."

Primitive gates all PASS. **T5 CLOSED 2026-08-20 — the consumer adoption LANDED**:
riir-clippy Bench 040 / Plan 016 shipped the input-scope gate at the `heal()`
seam (`ScopeModel` over the domain corpus vs a canonical 18-doc non-Rust
out-corpus; the 2D complement of the Issue 030 rule-coverage gate). Measured:
OOS inputs declined **8/8** vs **8/8 SERVED** un-gated (the demonstrated
defend-wrong bug with seeded memory — the production state); in-distribution
healing **bit-identical** over the full 40-fixture corpus (the haircut
saturates to exactly `1.0f32`); **529 ns/input** (release); steady-state
alloc delta **−1** (adds zero); the L4 path never sees garbage (0 calls vs
1 — saves the ~48 s GPU call). **κ/θ re-pinned by the consumer from its
measured gap** (θ = 0 — in-margin 7.6 bits / out-margin 24.4 bits; κ = 4.5 —
saturation 34.3 ≥ 16.6 at the worst fixture), discharging this bench's
"consumers re-pin per their benches" caveat. Honest scope: Rust-vs-not-Rust
garbage only — cross-domain Rust stays out (Issue 020's ~70% lexical
ceiling). ⇒ **`contrastive_scope` PROMOTED to katgpt-core `default`**
(2026-08-20). The riir-clippy consumer stays opt-in at its own feature level
(the `l4_lora` install-time precedent).

## Substrate check (substrate-first skill)

Searched vocabulary: `log_odds|log_ratio|naive_bayes|llr|contrastive|score_table`. Found: `soft_bayesian_omega` (similarity_inference — private match/mismatch LLR, different quantity); `occupancy::LogRatioClass` (KL-projection density-ratio estimator — different mechanism); BM25 (riir-neuron-db — single-corpus TF-IDF ranking, not two-corpus contrast); all other "contrastive" hits are latent-direction contrast (TILR/CNA/MAG). **Decision: build new** — the issue's "no analog" claim re-confirmed with vocabulary translation.

## Run

```bash
# DEFAULT-ON since the 2026-08-20 promotion — no feature flag needed:
cargo test -p katgpt-core --lib contrastive_scope::  # 7 tests (1 timing test ignored in debug)
cargo test --release -p katgpt-core --lib contrastive_scope::  # incl. the µs timing gate
cargo test -p katgpt-core --test contrastive_scope_alloc_check --release
```
