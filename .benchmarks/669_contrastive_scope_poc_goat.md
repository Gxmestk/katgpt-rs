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

Primitive gates all PASS. **No consumer adopted this session** — the natural pairings (riir-clippy L4: rule-coverage × input-scope 2D gate over the Issue 017 in/adversarial fixture corpora; riir-ai engram gates: shard-corpus vs OOD-zone embeddings) are recorded as the adoption paths. Per the rule: **not promoted; issue closed with the POC verdict recorded here** (a shipped opt-in primitive awaiting its consumer — not a negative on the mechanism, whose toy-corpus gates all passed).

## Substrate check (substrate-first skill)

Searched vocabulary: `log_odds|log_ratio|naive_bayes|llr|contrastive|score_table`. Found: `soft_bayesian_omega` (similarity_inference — private match/mismatch LLR, different quantity); `occupancy::LogRatioClass` (KL-projection density-ratio estimator — different mechanism); BM25 (riir-neuron-db — single-corpus TF-IDF ranking, not two-corpus contrast); all other "contrastive" hits are latent-direction contrast (TILR/CNA/MAG). **Decision: build new** — the issue's "no analog" claim re-confirmed with vocabulary translation.

## Run

```bash
cargo test -p katgpt-core --features contrastive_scope --lib contrastive_scope::   # 8 tests
cargo test --release -p katgpt-core --features contrastive_scope --lib t2_scope_score_us -- --nocapture
cargo test -p katgpt-core --features contrastive_scope --test contrastive_scope_alloc_check --release
```
