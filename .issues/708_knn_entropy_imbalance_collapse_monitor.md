# Issue 708: kNN differential-entropy estimator + two-channel imbalance collapse monitor (GenFirst extractions)

**Status:** Open — P2 remains (monitor + lead-time GOAT). **P1 DONE 2026-09-02** (this commit): `data_probe/entropy.rs` behind opt-in `knn_entropy` (katgpt-core + root shim + feature forwards); Kozachenko–Leonenko import with calibration/monotone/collapse/determinism unit gates ALL PASS (10/10); clippy 0; default build untouched (1992/0). P2 novelty TBD — see §P2.

## Source

Research 437 (riir-train/.research/437_GenFirst_Generation_Before_Reconstruction.md; arXiv:2608.29335 "GenFirst"). The paper's collapse analysis is two-channel: `D_KL(q‖p₀) = E_q[−log p₀(z)] − H(q(z|x))` — collapse is an IMBALANCE between concentration pressure and spread, not "entropy below τ". Transferring that law to runtime (modelless) surfaces needs two primitives neither of which ships.

## P1 — kNN differential-entropy estimator (the missing substrate) — DONE 2026-09-02

- Codebase prior art: ZERO (grep `kozachenko|knn_entropy|differential_entropy|entropy_estimate` over katgpt-rs = 0 hits). Shipped neighbors: `effective_rank` (katgpt-core `data_probe/geometry.rs`), `avg_cosine_similarity` (anisotropy), `gaussianity_probe`, `spectral_flatness` (riir-neuron-db) — dispersion proxies only, no uncertainty/entropy axis.
- Published prior art: classic (Kozachenko–Leonenko 1987; sklearn-shipped). This is an IMPORT, not an invention.
- Host: `katgpt-rs/crates/katgpt-core/src/data_probe/entropy.rs` (LANDED), beside `gaussianity.rs` (scratch + f64 + determinism conventions reused).
- Form: Kozachenko–Leonenko `Ĥ ≈ ψ(n) − ψ(k) + ln c_d + (d/n) Σᵢ ln εₖ(i)` over point-latent populations (belief states, emotion fields, span embeddings, fix trajectories). Brute-force kNN by design at offline N (O(n²·d), O(k) bounded max-heap scratch — sift-up fill + replace-root). Duplicates ⇒ −∞ (the collapse signal, honestly propagated; the P2 monitor interprets it, this fn does not clamp).
- Gate status: G1 closed-form calibration (isotropic Gaussian d4 ±0.35 / d8 ±0.9 nats) + monotone under shrink (σ 2→1→0.25, each arm near its own closed form) + planted rank-1 collapse trip (−27.6-nat drop, ≥15-nat margin) — ALL PASS; digamma (ψ(1)/ψ(0.5)/ψ(6) at 1e-12) + ln-unit-ball-volume (c₁..c₅ closed forms at 1e-12) known-answer pins PASS; G2 offline-scale smoke (n=2048, d=16, correctness-at-scale) PASS in 0.42s total suite; G4 zero-alloc by construction (scratch fixed at `new`); determinism ×3 bit-identical PASS. 10/10 tests, clippy 0, default build 1992/0 unchanged. Full GOAT bench deferred to consumer time per the no-default-consumer rule (the unit gates above are the filing's gate list, all green).

## P2 — Two-channel imbalance collapse monitor (the consumer)

- Codebase prior art: shipped detectors are ABSOLUTE-threshold only — `EntropyCollapse` (katgpt-core cgsp, `entropy < tau_low`), `S2FCollapseDetector` (katgpt-pruners, hesitation-token counting), erank/gaussianity floors (edge_lora dist guard, riir-train Bench 494). None observe the balance pair.
- Form: channel A = spread/entropy (P1), channel B = concentration/fit pressure; alarm on imbalance (A falling while B rising), not `h < τ_low` alone. The paper-derived delta is balance-vs-absolute.
- Hosts: `katgpt-pruners/src/collapse_detector.rs` + the detector trait in `katgpt-core/src/traits/mod.rs` (already carries collapse-threshold state).
- GOAT gate: planted-collapse severity sweep on the existing bench_681 fixture populations (`gaussian_population`, `bimodal_axis_population`, `lattice_population_d8`); **detection lead time** — imbalance flags degradation N cycles before the absolute threshold fires, monotone in severity. Falsifiable: lead time ≈ 0 ⇒ the imbalance framing adds nothing over τ_low ⇒ kill.
- Game-context reframe: per-NPC belief/emotion populations (cgsp) get lead-time collapse warnings before a personality collapses into exploitation; consolidation (riir-neuron-db Raven/δ-Mem) gets the same pair for coverage-vs-fit balance.

## Constraints

- The paper's LOSS-level entropy mechanism does NOT transfer (runtime surfaces have no gradients; our collapses are behavioral, not posterior-collapse). Only the observable-balance law transfers — causal justification re-derived per surface (Research 437 §2 transfer boundary).
- Do NOT build closed-loop adaptive weight controllers anywhere off the back of this paper — its own Table 2 shows PI-adaptive collapse outright; event-triggered corrections only.

## Related

- Research 437; riir-train Issues 500/501; Bench 494 (dist guard); bench_681 fixture populations; katgpt-rs Research 502 (lossy-surface consumer-metric rule — the same consumer-first discipline the monitor's lead-time gate encodes)
