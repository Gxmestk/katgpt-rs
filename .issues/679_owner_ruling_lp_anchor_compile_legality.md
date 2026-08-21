# Issue 679: OWNER RULING — LP-anchor compile legality for spectral pencil gates (Research 495)

> Source: [Research 495](../.research/495_Spectral_Neuron_Affine_Pencil_Shape_Gates.md) §3.1 P2. A decision request, not yet an implementation. The No-GD panel flagged this as the one item needing an explicit ruling; it must not silently drift into "obviously fine".

## The question

Does **LP anchor interpolation on the commuting pencil subclass** qualify as modelless-construction under the modelless-first mandate?

The route: constrain the pencil to simultaneously-diagonalizable (commuting) matrices, where `f(x) = λk(A0 + ΣxᵢAᵢ)` degenerates to the **k-th smallest of d affine forms** (an order statistic — verified coverage: the commuting case IS maxout/order-statistics, Goodfellow 2013 / Rennie 2014). Fitting anchor points `(x_j, y_j)` then means choosing the d affine coefficients so the k-th order statistic passes through them — a **quantile-curve feasibility/interpolation LP**. Given a pinned solver, same anchors → same matrices bit-wise.

**Argument FOR legality** (the panel's position): it is a *deterministic construction from data* — same legal class as the mandate's sanctioned "deterministically constructed LoRA overlay" (`raw/lora hot-swap`, no gradient descent, same input → same output bit-wise). An LP solve is not gradient descent; there is no iterative weight mutation, no loss landscape, no convergence variance.

**Argument AGAINST**: an LP solver is still an *optimizer*; the spirit of the mandate is "closed-form or provable construction", and "offline solver output" sits closer to the riir-train boundary than anything else admitted so far. Precedent tension: `ConstraintPruner`-family substrates do constraint reasoning, but none ship an LP solve.

## Routes context (the compile ladder this sits in)

1. Designer knobs (k, PSD/NSD signs, α/ε scales) — **ships in Issue 676 T7**, unambiguous.
2. Rank-one directions over BLAKE3 direction vectors — **ships in Issue 676 T7**, unambiguous.
3. **LP-anchor interpolation (commuting subclass)** — this ruling.
4. Seeded property-test search (sample 676 constructions until one passes a property test; deterministic given seed + test) — unambiguous (same seed → same bytes).

## Tasks

- [ ] D1 Owner ruling: legal-as-modelless-construction / riir-train-territory / reject outright. One line, recorded here + mirrored into Research 495 §3.1.
- [ ] D2 If LEGAL: implement offline-only (never hot-path) — LP feasibility against anchors, pinned solver + determinism bit-check gate, anchors-matched-exactly property test; document limits (commuting subclass only ⇒ piecewise-linear expressivity; the nonlinearity dial collapses to the order-statistic kink structure).
- [ ] D3 If NOT LEGAL: no code; note the boundary in Research 495 §3.1 with the ruling, and route any real fitting demand to riir-train 472 (trained heads).

## Non-goals

No fitting of non-commuting pencils by any solver (that is unambiguously riir-train 472 territory). No hot-path LP anything, under any ruling.
