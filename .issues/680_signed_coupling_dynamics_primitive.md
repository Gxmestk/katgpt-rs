# Issue 680: signed_coupling_dynamics primitive — signed-graph opinion propagation + crowd order parameters

**Research:** [katgpt-rs/.research/497_Signed_Coupling_Opinion_Phase_Forecast.md](../.research/497_Signed_Coupling_Opinion_Phase_Forecast.md)
**Source:** [arXiv:2608.16578](https://arxiv.org/abs/2608.16578) "Physics of Agents" — Glauber/Ising update on signed social graphs
**Target:** `katgpt-rs/crates/katgpt-core/src/signed_coupling.rs` + Cargo feature `signed_coupling_dynamics` (opt-in)
**Status:** Open
**Priority:** P2 — blocked on nothing; substrate exists (CLR set attention sibling)

---

## Problem

Research 497 distills arXiv:2608.16578: crowd opinion dynamics follow `P(s_i=+1) = σ(β⁺Σ J⁺_ij s_j + β⁻ Σ J⁻_ij s_j + β₀ Σ|J_ij| s_j + g_i)` — a sigmoid of a signed, tie-typed weighted neighbor sum, plus crowd order parameters (net opinion n = mean(s), conviction c = mean(s²), susceptibility χ = N·Var_t(|n|) whose peak locates the critical social temperature).

The substrate audit (Research 497 §2) found: the σ(gated weighted sum) shape ships (CLR set attention — unsigned), a 2-channel tie-typed fear propagation ships (swarm threat-kick + rank-gated aura), but the **signed 3-coupling kernel, `mean(s²)` conviction reducer, and χ susceptibility accumulator ship nowhere**. No parallel system should be built — this is one small kernel + two reducers over the CLR substrate pattern.

## Tasks

- [ ] T1: `SignedGraph` row-compressed storage (J⁺/J⁻ neighbor lists, u32 index + no heap after construction) + `Couplings { beta_plus, beta_minus, beta_zero }` with paper-fitted defaults (β⁺ 0.9–2.4, β⁻ 0.2–1.1, β₀ 0.6–1.0)
- [ ] T2: `signed_coupling_update_into` — O(edges), zero-alloc, writes caller scratch; the informed variant (`κ_j` per-neighbor indicator splitting β_T/β_F, the paper's 5-coupling truth asymmetry) as a sibling fn, not a flag
- [ ] T3: order-parameter reducers — `net_opinion` (mean), `crowd_conviction` (mean of squares — NEW), `SusceptibilityAccumulator` (running Var_t(|n|)); T_c location ships as an **offline example/bench only** (41-point log sweep is not a runtime path)
- [ ] T4: feature flag `signed_coupling_dynamics = []` (opt-in) — NOT in default
- [ ] T5: GOAT gate — G1: seeded rollout reproduces the paper's three regimes (indifference/polarization/consensus) on synthetic signed graphs (random + square lattice + a low-rank frustrated graph); G2: latency vs hand-rolled baseline at N=32/256/1024 (Plasma tier target); G3: default-features untouched; G4: alloc-free steady state (CountingAllocator)
- [ ] T6: doc note pinning the "conviction" vocabulary split — `crowd_conviction` (this crate, order parameter mean(s²)) vs Sheaf-ADMM `conviction` (riir-agents, per-dim consensus resistance). Both greps must land on the disambiguation note.

## Non-goals

- No riir-ai consumer wiring in this issue (swarm/salience/MCTS fusion is downstream; run goat-audit before any riir-ai plan consumes this).
- No calibrated-forecasting claim — the kernel is a dynamics rule; any prediction-quality claim requires the conformal floor (Report the Floor) + defend-wrong PoC.
- No continuous tanh relaxation, no Potts/vector-opinion extension (defer, Research 497 §11).

## Acceptance

GOAT G1–G4 pass behind the feature flag; stays opt-in until a production consumer promotes it (the CLR precedent).
