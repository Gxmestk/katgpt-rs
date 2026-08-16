# Issue 659 — Modelless bigram Markov head: the missing piece of hybrid DSpark (Bonsai)

**Status:** Open
**Opened:** 2026-08-16
**Owner:** katgpt-rs
**Related Research:** 316 (DSpark), 407 (Trees from Marginals)
**Related Plans:** 339 (Hardware-Aware Prefix Scheduler — SHIPPED, opt-in)
**Related Benchmarks:** 656 (MTP Metal batch-width floor)
**Consumer:** Ternary-Bonsai-27B (`riir-train/data/Ternary-Bonsai-27B-dspark-Q4_1.gguf`)

## What is already shipped

Research 316 states the modelless hybrid DSpark is feasible today from parts
that already exist:

| component | status |
|---|---|
| DFlash factorized drafter | shipped |
| Bebop entropy confidence (`AcceptanceForecast`) | shipped |
| Hardware-Aware Prefix Scheduler | shipped (Plan 339, opt-in, GOAT G1–G5 PASS) |
| **Sequential Markov head** | **NOT shipped** |

The Markov head is the one gap, and Research 316 §3.5 path 2 records it as
**modelless-constructable from corpus bigram statistics** (only the RNN-head
variant needs training → riir-train).

## Why this matters for Metal specifically

Benchmark 656 separated two distinct Apple-Silicon failure modes. Mode 2 is
structural: a **separate multi-layer drafter** must run its own forward every
step, and at batch-1 there is nothing to amortise it against. That is exactly
why `riir-train/README.md:248` records DSpark as *"on Apple Silicon batch-1
verification doesn't amortize the drafter; on 4090 it gives the 1.34× speedup."*

A bigram Markov head is a **table lookup, not a forward pass**. It does not incur
mode 2 at all. So the modelless hybrid could plausibly win on Metal where the
PrismML-trained 6-layer DSpark drafter loses — which is the whole reason Bonsai
wants this.

## Design sketch (modelless)

- Build `P(next | prev)` from corpus bigram counts — deterministic, no training.
- Store top-`m` successors per token (a compact CSR-style table), not the dense
  `V × V` matrix.
- Emit per-position marginals so the output drops straight into
  `build_dd_tree(marginals: &[&[f32]], config)` — the existing seam, which has
  zero `dflash` coupling in production code.
- Compose with the shipped Bebop confidence + Plan 339 scheduler.

## GOAT gate

- **G1** — lossless: verified output identical to autoregressive decode
  (speculative decoding must not change the distribution).
- **G2** — acceptance rate vs the DFlash baseline at equal draft depth.
- **G3** — wall-clock on **Metal** (the mode-2 claim) *and* CUDA. Must beat
  baseline on Metal, or the headline motivation is unproven.
- **G4** — alloc-free hot path; table built once at load.
- **G5** — table memory bounded (report bytes at Bonsai's vocabulary).

## Tasks

- [ ] Bigram table builder from corpus counts (deterministic).
- [ ] Top-`m` successor storage + marginal emission.
- [ ] Wire into `build_dd_tree`.
- [ ] Bench on Bonsai; Metal and 4090.
- [ ] Note: Plan 339's scheduler GOAT is *vacuous* today ("katgpt-rs default is
      single-request, so the gate is vacuous without a multi-request batch
      caller"). This issue does not fix that; a real multi-request caller is
      separate work.
