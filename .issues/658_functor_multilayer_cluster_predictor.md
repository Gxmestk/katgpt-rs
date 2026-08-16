# Issue 658 — Multi-layer / FUNCATTN cluster predictor for the clustered LM head

**Status:** **MOOT — closed without implementing (2026-08-16).** The quality gap
this was to close no longer exists. See §Outcome.
**Opened:** 2026-08-16
**Closed:** 2026-08-16
**Owner:** katgpt-rs
**Related:** Plan 574 T7, Issue 657 (resolved), Research 026 (Gemma 4 MTP),
`.benchmarks/658_clustered_lm_head_admissible_goat.md`

## Outcome — no residual gap to close

This issue was ordered *after* Issue 657 on the reasoning quoted below: "more
layers cannot repair a wrong objective." Benchmark 658 found the objective was
never wrong. Plan 574's recall failure was a degenerate k-means seeding
(strided init at `stride = vocab/k` drew every centre from two planted groups);
with D² seeding the **existing single-layer centroid score reaches argmax recall
1.0000 at a 2% active budget**, and the admissible bound proves the exact argmax
after touching 7.30% of the vocabulary.

A deeper predictor would be optimising a metric that is already saturated. The
premise — "stage 1 sees too little of the network" — was never tested against a
correctly-seeded map, and once it is, there is nothing left to explain.

**Reopen only if** Issue 662 (real checkpoint) shows recall well below 1.0 on
real weights with D² seeding. In that case the information-scarcity hypothesis
becomes live again for the first time, and this issue's substrate list
(`funcattn`, `cross_resolution_transport`, `project_target_activation`) is still
the right place to start. Until then it is speculation about a solved problem.

The current blockers on Plan 574 are **cost**, not quality: Issue 661 (serial
stage 2) and Issue 662 (real checkpoint).

---

## Original proposal (preserved)
**Substrate:** `funcattn` (DEFAULT-ON), `funcattn_structured_basis` (DEFAULT-ON),
`cross_resolution_transport` (Plan 310, DEFAULT-ON), `mtp.rs::project_target_activation`

## Idea

Stage-1 cluster selection currently sees **only the final hidden state**. Feed it
more of the network instead, transporting intermediate-layer representations
into the LM-head space with the closed-form operators already shipped:

- **FUNCATTN** — closed-form Tikhonov spectral transport (arXiv:2605.31559),
  modelless, already default-on.
- **`cross_resolution_transport`** (Plan 310) — default-on, G1 mean cos 0.8944,
  G2-A rank preservation 0.9300. Transports between resolutions.
- **Gemma 4 precedent** — Research 026's "target activations" mechanism feeds the
  target's hidden state into the drafter; `project_target_activation()` already
  implements the projection with a learned-or-truncate/pad fallback.

## Why this is the *second* lever, not the first

Benchmark 657 shows the selector admitting 102 of 252 clusters at a 25% budget
and still missing the argmax ~32% of the time. That points at the **scoring
objective** (mean logit vs max logit — Issue 657), not at information scarcity.
More layers cannot repair a wrong objective, and the radius bound in Issue 657
is both cheaper and *provably admissible*.

> **Refuted (Benchmark 658).** The ordering was right for the wrong reason. It
> was not a scoring defect *or* information scarcity — it was the clustering
> itself. Both this issue and Issue 657 misread the same evidence, because both
> took the cluster map as a given and asked what to do with it.

Do 657 first. If recall still falls short of 0.99 at an acceptable active
fraction **after** the bound lands, the residual gap is genuinely
informational and this is the right next move.

## Tasks

- [ ] Land Issue 657; re-measure. Only proceed if a real gap remains.
- [ ] Transport layer-`L` hidden states into LM-head space via FUNCATTN.
- [ ] Score clusters on the concatenated/transported signal.
- [ ] Gate: recall improvement must exceed the added stage-1 cost — the whole
      point of clustering is to be cheaper than the full head.
- [ ] Keep modelless: `project_target_activation`'s learned `mtp_proj` path is a
      trained artifact. Use the closed-form transport, not a trained projection,
      or the primitive loses its modelless status.
