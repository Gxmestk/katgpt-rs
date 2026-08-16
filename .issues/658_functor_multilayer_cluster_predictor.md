# Issue 658 — Multi-layer / FUNCATTN cluster predictor for the clustered LM head

**Status:** Open (second lever — try Issue 657 first)
**Opened:** 2026-08-16
**Owner:** katgpt-rs
**Related:** Plan 574, Issue 657, Research 026 (Gemma 4 MTP)
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
