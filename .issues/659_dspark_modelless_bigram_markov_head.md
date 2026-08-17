# Issue 659 — Modelless bigram Markov head: the missing piece of hybrid DSpark (Bonsai)

**Status:** Open
**Opened:** 2026-08-16
**Owner:** katgpt-rs
**Related Research:** 316 (DSpark), 407 (Trees from Marginals)
**Related Plans:** 339 (Hardware-Aware Prefix Scheduler — SHIPPED, opt-in)
**Related Benchmarks:** 656 (MTP Metal batch-width floor)
**Consumer:** Ternary-Bonsai-27B (`riir-train/data/Ternary-Bonsai-27B-dspark-Q4_1.gguf`)

## CORRECTION (2026-08-16, after riir-ai substrate check)

This issue was originally framed as "the missing piece of hybrid DSpark". That
framing is **wrong**. riir-ai Plan 528 §81 (the Bonsai owner, Active, Phase 0)
states it plainly:

> Bonsai **SHIPS with DSpark** — a 6-layer block-denoising speculative drafter
> (1.34× CUDA decode speedup). DSpark IS the DFlash pattern. **Prism-ML trained
> it; we consume it.** The DFlash training gap from Plan 332 is closed for the
> Bonsai path by DSpark — **we don't need to train a drafter.** On Apple Silicon
> the DSpark drafter is not enabled by default (batch-1 verification doesn't
> amortize), but on 4090 it gives the 1.34× speedup.

So the drafter gap is **already closed** for Bonsai on CUDA. The real, narrower
gap is: **Bonsai has no working drafter on Apple Silicon**, because DSpark's
6-layer forward does not amortise at batch-1 (Bench 656 failure mode 2).

**Revised scope:** a bigram Markov head is not "the missing DSpark piece" — it is
a *Metal-viable alternative drafter* for the case where DSpark is switched off.
It is a table lookup rather than a forward pass, so it does not incur mode 2.

**Ownership:** Bonsai is riir-ai's (Plan 528, 7 open tasks). The *primitive*
belongs in katgpt-rs (which owns `dd_tree`, `dflash`, `prefix_scheduler`); the
*Bonsai consumer + gate* belongs in riir-ai. Coordinate before implementing.

**Also relevant — riir-ai Issue 708 (open, filed 2026-08-16):**
`forward_dflash.rs` (1,345 LOC) is **NON-FUNCTIONAL and wrong-math** (H1: stale
`uniform_qkv` makes every Phase-A dispatch read `pos = seq_len-1`), and
"presents as working the moment it is enabled — the backward tests pass on
garbage output". This is the concrete reason the dflash investment appears
unused. Fixing H1–H4 may be higher value than adding a new drafter.

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

- [x] Bigram table builder from corpus counts (deterministic). — `BigramMarkovBuilder` (packed-u64 sort + two-pointer-pass; `(count desc, next asc)` top-m; bit-identical rebuilds, brute-force-reference-pinned).
- [x] Top-`m` successor storage + marginal emission. — `BigramMarkovTable` (CSR) + `bigram_predict`/`BigramMarginalBuffer`: zero-alloc steady state, O(steps × top_m) touched-reset sparse writes, greedy-chain conditioning, zero-row fallback for unseen prevs (the seam skips `prob ≤ 0` — unseen proposes nothing).
- [x] Wire into `build_dd_tree`. — `bigram_build_tree` (emits `config.draft_lookahead` marginals → `build_dd_tree` seam).
- [-] Bench on Bonsai; Metal and 4090. — **Primitive gate landed (Bench 663, 2026-08-17)**: 181 ns/call (23 ns/step) at Bonsai scale on M3 release — ~5,600× under a 6-layer drafter forward per step (the mode-2 avoidance measured); 17 MB worst-case table vs 268 MB low-rank. **Deferred**: the consumer gate (acceptance rate at equal draft depth + wall-clock on Metal AND 4090 against the Bonsai target) — belongs to the riir-ai Bonsai consumer (Plan 528), and the 4090 is occupied by a sibling's p336 run. The `bigram_markov` feature stays opt-in until this gate passes.

> Note: Plan 339's scheduler GOAT is *vacuous* today ("katgpt-rs default is
> single-request, so the gate is vacuous without a multi-request batch
> caller"). This issue does not fix that; a real multi-request caller is
> separate work.

## Progress log

- **2026-08-17 (T1–T3 + primitive gate)**: substrate-first check run — no
  existing bigram/n-gram/successor-table substrate in any of the 7 repos
  (`bigram`/`markov`/`ngram`/`successor` variants grepped; `belief_drafter`
  is the trained-MLP sibling, not a conflict). Module landed in
  `katgpt-speculative` beside `dflash.rs`/`dd_tree/` per the ownership note;
  feature `bigram_markov = []` (zero deps). The issue's Issue-708 caveat is
  moot — 708 resolved (`42b564759`). The batched width-N Metal forward
  (Bench 662) landed the verification side, so the remaining gap is exactly
  the consumer gate above. Record: [Bench 663](../.benchmarks/663_bigram_markov_head_primitive.md).
