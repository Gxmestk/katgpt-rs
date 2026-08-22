# Format Templates — Supporting Reference for the Research Skill

> **When to read this file:** when creating a new `.research/NNN_*.md` note or `.plans/NNN_*.md` plan. Canonical examples to follow, not copy verbatim — adapt the sections to your paper.

## Research note format

Canonical example: `katgpt-rs/.research/238_LoRA_Muon_Spectral_Low_Rank_Manifold.md`

```markdown
# Research NNN: {Title}

> **Source:** <paper title + arxiv link + authors + date>
> **Date:** YYYY-MM-DD
> **Status:** Active | Done | Shelved
> **Related Research:** NNN (short note), ...
> **Related Plans:** NNN (short note), ...
> **Cross-ref (riir-ai / riir-chain / riir-neuron-db):** Research NNN, Plan NNN   ← only if cross-repo
> **Classification:** Public | Private   ← katgpt-rs notes are always Public

---

## TL;DR

<2-4 sentences: the distilled primitive, why it matters here, what it unblocks>

**Distilled for katgpt-rs (modelless, inference-time):**
<the transferable insight, stripped of training setup>

---

## 1. Paper Core Findings
...
## 2. Distillation
...
## 3. Verdict

**Tiers (high → low):**

| Tier | Criteria | Routing |
|------|----------|--------|
| **Super-GOAT** | Novel mechanism (no prior art) + new capability class + product selling point + force multiplier (≥2 pillars). Creates a moat. | Open primitive → katgpt-rs. **Architectural guide → riir-ai/.research/ OR riir-chain/.research/ OR riir-neuron-db/.research/**. Plans → appropriate repo(s). |
| **GOAT** | Provable gain (latency/quality/security) over existing approach, but not a new class of capability. Promotes to default if it wins. | Plan + implement. Feature flag + benchmark. |
| **Gain** | Incremental improvement, useful but not headline-worthy. | Plan only, behind feature flag. |
| **Pass** | Mechanism already ships OR not relevant OR training-only AND genuinely out of scope. | **No new files.** Verdict + one-line reason + closest shipped cousin, in conversation only. |

**One-line reasoning required for each verdict.** For Super-GOAT: state the selling point explicitly.

**After the tier verdict, run the MOAT gate per domain (main skill §1.6)** — a tier verdict without a domain-fit check can land a great primitive in the wrong repo and dilute the moat.
```

**File naming:** `{NNN}_{Short_Title_with_Underscores}.md` where NNN is the next free number (zero-padded to 3 digits, e.g. `239_`, `240_`). Check the folder first — numbers may be non-contiguous; pick the next free slot. **Numbers are monotonic and never reused** — even after a file is removed per the noise-reduction rule. Check `.highwater` if it exists.

## Plan format

Canonical example: `katgpt-rs/.plans/271_attention_matching_compaction.md`

```markdown
# Plan NNN: {Title}

**Date:** YYYY-MM-DD
**Research:** [katgpt-rs/.research/NNN_*.md](../.research/NNN_*.md)
**Source paper:** [arxiv ID.NNN](https://arxiv.org/abs/ID) — <short cite>
**Target:** `katgpt-rs/src/{module}/` (new module) + Cargo feature `{feature_name}`
**Status:** Active — Phase N {state}

---

## Goal

<one paragraph: what ships, what it enables, GOAT gate>

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [ ] **T1.1** <concrete, verifiable task>
- [ ] **T1.2** ...
```

Use `## Phase N` sections with `- [ ]` per task (mark `- [x]` when done, `- [-]` for deferred).

**GOAT gate rule** (AGENTS.md): every plan that introduces a new technique must have a feature flag and a benchmark proving the gain before promoting to default. Demote the loser if the new technique wins.

**UQ-bearing primitive GOAT gate extension (the "Report the Floor" rule, Research 322):** Any primitive claiming a probability distribution, predictive interval, quantile, coverage guarantee, or calibrated uncertainty MUST benchmark against the conformal-naive floor (`ConformalIntervalCalibrator<SeasonalNaiveForecaster>`, Plan 340 with `m=1`). If it cannot beat the floor on CRPS / empirical coverage / Winkler score, the GOAT gate FAILS. Existing UQ primitives (BoMSampler Plan 281, Sleep-Time Plan 334, Best-Belief Plan 336, KARC+overlay) are grandfathered but must include the floor at their next re-gate. The floor shipped in Plan 340 Phase 1 (2026-06-30); the rule is now enforceable. See `.benchmarks/010_report_the_floor_consolidated.md`.
