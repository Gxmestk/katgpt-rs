# Plan 585: Usage-Rate (Mass/Age) KV Eviction Primitive + Generation-Runaway Canary

**Date:** 2026-08-31
**Status:** DONE (2026-09-02) — Phases 1–3 landed; **MIXED GOAT verdict, opt-in** (Bench 697: G1–G4 PASS, G8 regime-bounded — 2–4× raw-H2O recall at cap ≥ 32, one honest miss at the 8%-budget extreme; no consumer = no promotion). Phase 4 stays consumer-pull-gated ([-] below). Record: [`.benchmarks/697_usage_rate_eviction_goat.md`](../.benchmarks/697_usage_rate_eviction_goat.md).
**Research:** [katgpt-rs/.research/523_H2O_Norm_Age_Normalized_KV_Eviction.md](../.research/523_H2O_Norm_Age_Normalized_KV_Eviction.md)
**Source paper:** [arXiv:2608.19920](https://arxiv.org/abs/2608.19920) — "Learning how to Forget" (Seeger et al., AWS, 2026)
**Target:** `katgpt-rs/crates/katgpt-core/src/kv_eviction/` (new module) + Cargo feature `usage_rate_eviction`
**Track:** PRIMARY (modelless; serving-envelope fit — the score runs at eviction time inside the hot path). The SECONDARY training track is riir-train Plan 367.

---

## Goal

Distill the paper's normalized H2O score into a pure, leaf-clean katgpt-core primitive: per-row usage-rate eviction scoring (`mass / max(1, age)`) over caller-supplied attention-mass increments, with pinned-sink exclusion and per-(b,h) selection by construction — plus the **R/p128 generation-runaway canary**, the output-length diagnostic that gates any lossy KV policy's promotion (the generation-side complement of the Issue 750 lossy-surface rule). GOAT gate: at matched KV budget on a planted age-bias fixture, mass/age retains the hot row raw-H2O evicts, with O(1)/row/step updates and zero steady-state allocation. Not UQ-bearing (no distributions claimed) — the conformal-floor rule does not apply.

House pattern: the primitive consumes caller-supplied observational signal (`suspect_indices(attention_mass, …)` precedent) — katgpt-core stays leaf-clean; mass producers are consumer-side (riir-gpu kernel byproduct → riir-ai Issue 836; HLA recurrent-state probe → Phase 4).

## Phase 1 — Primitive (CORE)

### Tasks

- [x] **T1.1** `crates/katgpt-core/src/kv_eviction/mod.rs` behind `usage_rate_eviction = []` (opt-in): `UsageScoreTable` — per-row `cum_mass: f32` + `admission_tick: u64`, fixed-capacity (caller-owned buffers, zero-alloc; `Vec::with_capacity` once, `clear()`+reuse). *(landed 2026-09-02)*
- [x] **T1.2** `observe(&mut row, mass_increment, tick)` — O(1) accumulate; `score(row, tick) -> f32` = `cum_mass / max(1, tick - admission_tick)` (min-age 1: fresh rows score at their mass, never ÷0; NaN guard: non-finite increment → ignore + debug_assert). *(landed)*
- [x] **T1.3** `select_evict(scores, k, pinned: &[bool]) -> Vec<usize>` — lowest-k scores among unpinned rows, deterministic tie-break by index (ascending) via the `float_order` total-order comparators (`float_order.rs`, NaN-safe — the partial_cmp-unwrap_or intransitivity fix); per-(b,h) by construction (caller slices per head — no cross-head reduction anywhere). *(landed as `select_evict` + zero-alloc `select_evict_into` core — `out` doubles as the workspace; output in eviction-priority order, lowest score first)*
- [x] **T1.4** Property tests: monotone in mass, anti-monotone in age, pinned rows never selected, β=0-pin is a no-op on selection, determinism (bit-identical across runs). *(20 module tests)*
- [x] **T1.5** Reference-parity test vs a naive recomputing implementation on LCG-generated streams (bit-identical scores).

## Phase 2 — Generation-Runaway Canary

### Tasks

- [x] **T2.1** `kv_eviction::canary`: `RunawayStats::from_generations(output_lens: &[usize], target_lens: &[usize], cap: usize)` → `R_median` (output/target ratio), `p_cap` (fraction at cap). Pure fn, zero deps. *(landed; zero-target samples skipped)*
- [x] **T2.2** Encode the promotion rule as a documented fn `runaway_gate(stats, r_max: f32, p_cap_max: f32) -> bool` + doc: **any lossy KV policy (eviction/quantization/compaction) promoted to default MUST pass this gate on a sealed long-context eval** — extends the Issue 750 lossy-surface rule to the generation axis. *(landed; empty-eval FAILS the gate — the vacuity guard)*
- [x] **T2.3** Non-vacuity test: a planted over-eviction fixture must FAIL the gate (fails-before/passes-after — the tile-loop-gate lesson). *(module test + bench-level canary demo: ring@8 R=8.0 FAILS, mass_age@32 R=1.0 PASSES on identical thresholds)*

## Phase 3 — GOAT Bench (falsifiable)

### Tasks

- [x] **T3.1** Planted age-bias fixture: synthetic attention stream with an old-but-cold row (0.001/step × 1000) vs young-but-hot row (0.5/step × 2) at equal raw-H2O cumulative mass — raw-H2O must evict the hot row, mass/age must retain it. The bench is invalid unless this fixture fires. *(PASS: tie arm + strict arm both fire)*
- [x] **T3.2** Micro-GPT long-context recall at matched KV budget (Bench 313 micro-GPT precedent): policies {ring/lastrec β, raw-H2O, mass/age, mass/age+sink-pin, EGA-energy, EGA×usage fusion} — recall/accuracy + `RunawayStats` + eviction-count per policy. *(constructed induction-pair KV, drifted-Zipf workload; 6 policies × 4 caps × 32 seeds — mass_age 50/100/100% vs raw 24/41/51% at caps 32/48/64, miss at cap 16 recorded)*
- [x] **T3.3** Kendall-τ diagnostic: per-head vs batch-summed top-k disagreement over the streams (decides whether per-(b,h) bookkeeping pays; τ ≈ 1 on our workloads ⇒ keep per-head anyway since it is free here, record τ for the kernel-side decision). *(τ = 0.689–0.748 — rankings DISAGREE materially, per-head bookkeeping matters)*
- [x] **T3.4** Gates: G1 determinism (bit-identical double-run); G2 O(1)/row update (criterion, update path < 10ns/row target); G3 default-features no-regression (module fully gated); G4 zero steady-state allocs (TrackingAllocator); **G8 mass/age ≥ raw-H2O recall at matched budget on T3.1+T3.2** — if it loses, keep the negative-result artifact (Bench 697 precedent) and demote. *(G1 PASS; G2 1.22 ns/row PASS; G3 PASS; G4 PASS — after the gate caught a live 1 alloc/step; G8 MIXED: PASS at cap ≥ 32, FAIL at the 8%-budget extreme — regime boundary recorded, primitive NOT demoted since opt-in with no consumer is the standing state)*
- [x] **T3.5** Write `.benchmarks/NNN_usage_rate_eviction_goat.md`; per-stack ledger: slot = KV/eviction; promotion decision (default vs opt-in) per gate outcome + consumer presence. *(Bench 697; decision: OPT-IN — no consumer + regime-bounded G8; promotion re-gate = Issue 836 consumer + real-corpus re-run)*

## Phase 4 — Consumer Probes (pull-gated)

### Tasks

- [-] **T4.1** GPU byproduct kernel (summed attention weights alongside SDPA, cubecl + cudarc twins) → **riir-ai Issue 836** owns the wiring surface; pull-gated on this plan's GOAT pass.
- [-] **T4.2** HLA free-mass probe: linear-attention recurrent state may expose cumulative usage directly (no kernel work) — one probe bench before any kernel investment.
- [-] **T4.3** Replay-log → telemetry → Beta-LCB policy-variant selection (self-adaptive track): substrate exists (katgpt-core `rating`); no serving consumer for policy-variant selection until ≥2 policies run in production — reopen then.
- [-] **T4.4** Content-derived β (`smart_lastrec` regex-prefix variant): paper's own footnote 11 measured the general variant NO better than fixed prefix — fixed β stands; reopen only with a structure-tagged corpus showing fixed-β failure.

## Non-goals

- No runtime wiring in this repo (consumers live in riir-ai — Issue 836).
- No training anywhere (riir-train Plan 367 owns co-adaptation).
- Bonsai-GDN: no KV eviction on recurrent state — out of scope by architecture.
