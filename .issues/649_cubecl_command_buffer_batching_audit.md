# Issue 649 — CubeCL command-buffer batching: `CUBECL_WGPU_MAX_TASKS` audit

**Type:** optimization (measurement + programmatic default)
**Status:** RESOLVED for its own question — CubeCL already batches 32
dispatches/command-buffer; raising `tasks_max` to 1024 yields ~6% (below the
GOAT 1.10× gate); set as programmatic default for the free gain. Dispatch-fusion
track is COMPLETE on Metal.

**One open follow-up (2026-08-13):** the T3 default change altered submission
granularity 32× *after* riir-ai Bench 649 measured its contention baseline, so
(a) re-runs of that gate must pin `CUBECL_WGPU_MAX_TASKS` explicitly, and
(b) `tasks_max` is now an untested candidate modulator of the Issue 640
divergence. See "Interaction with riir-ai Issue 640" below.
**Filed:** 2026-08-13
**Found by:** Issue 648 closeout — "batch command buffer recording" was flagged
as "the biggest remaining optimization opportunity" but was never verified.

---

## The hypothesis (from the Issue 648 summary)

> Batch command buffer recording — a CubeCL/wgpu infrastructure change (batch
> multiple dispatches into one command buffer submission) would reduce
> per-dispatch overhead for ALL ~834 remaining dispatches.

This projected a large win from reducing per-dispatch overhead (~30µs each ×
~834 dispatches ≈ 25 ms/token of pure dispatch overhead).

## The reality: CubeCL already batches

CubeCL 0.11's wgpu backend (`cubecl-wgpu-0.11.0-pre.2/src/compute/stream.rs`)
uses a **single persistent `wgpu::CommandEncoder`** per stream. Each
`register_pipeline` call records a dispatch into the same compute pass. The
encoder is only `finish()`-ed + `queue.submit()`-ed when:

1. `flush_if_needed()` fires — `tasks_count >= tasks_max` (default **32**)
2. An explicit `flush()` / `sync()` / `read_one()` forces it

So with `tasks_max=32` and ~835 dispatches/token, CubeCL already batches into
~26 command-buffer submissions per token — NOT 835 separate submissions. The
"30µs per dispatch" figure was wrong for the batched case; the real per-dispatch
overhead within a compute pass is much lower (a `dispatch_workgroups()` call +
bind group setup).

## Measurement: `CUBECL_WGPU_MAX_TASKS` sweep

Hardware: M3 Max Metal, clean GPU state (no sibling processes). 30 decode
tokens, Ternary-Bonsai-27B.

| `tasks_max` | tok/s | ms/token | argmax | notes |
|---|---|---|---|---|
| 32 (default) | 22.30 | 44.8 | 198 | baseline |
| 1024 | 23.67 | 42.2 | 198 | +6.1% |

The gain is **~6%** — below the GOAT 1.10× gate. The entire forward pass
(~835 dispatches) fits into a single command buffer at `tasks_max=1024`, but
the saved overhead (~2.6 ms/token) is modest because CubeCL's 32-task batching
already captures most of the win.

## Tasks

- [x] **T1** — Audit CubeCL's wgpu server to understand the batching mechanism.
      **DONE** — single persistent encoder per stream; flushes at `tasks_max`.
- [x] **T2** — Measure `CUBECL_WGPU_MAX_TASKS` sweep (32 vs 1024). **DONE** —
      6% gain, below GOAT gate.
- [x] **T3** — Set `tasks_max=1024` as a programmatic default in
      `CubeCLContext::new()` (env var override still wins). **DONE** — free 6%
      with zero correctness risk.
- [x] **T4** — Verify correctness (argmax=198 unchanged) + no-regression.
      **DONE**.
- [x] **T5** — Document the finding: dispatch-fusion track is COMPLETE on Metal.
      The remaining bottleneck is GEMV kernel throughput (Issue 628 structural
      ceiling at ~45% roofline).

## Why the dispatch-fusion track is COMPLETE on Metal

Issues 642 + 645 + 648 saved 351 dispatches (8.47 → 22.30 tok/s = 2.64×
cumulative). The remaining ~835 dispatches are batched into ~26 command buffers
by CubeCL, and increasing the batch to 1 command buffer gives only 6% more.

The **real bottleneck** is the ternary GEMV kernel itself at ~45% of the 400
GB/s Metal roofline. Issue 628 closed ALL six non-hardware escape hatches for
this ceiling (interleaved layout 1.088×, base-3 trit 0.500×, shared-memory LUT
0.42×, subset-sum LUT 0.540×, fused GEMV+ResidualAdd 1.01×, CMMA analytically
refuted). The ceiling is **structural** to the ternary format on Metal — there
is no `__dp4a` equivalent (Bench 606 T3c: `dot4I8Packed` is 0.56-0.61× = 1.6-1.8×
SLOWER).

**Conclusion:** no further dispatch-level optimization on Metal will produce a
GOAT-worthy gain (≥1.10×). The path to higher Metal throughput would require a
fundamentally different weight format (not a kernel change) or accepting the
ceiling.

## Interaction with riir-ai Issue 640 (contention nondeterminism) — AUDITED 2026-08-13

Cross-checked this issue against riir-ai
[Issue 640](../../riir-ai/.issues/640_batched_rmsnorm_prefill_nondeterministic.md),
which established that a second process using the GPU makes wgpu/Metal produce
**bit-different results** for identical input — 4.05% → 50.00% divergence
(12.3×, χ²(1)=36.03, p≈2×10⁻⁹, riir-ai `.benchmarks/649_contention_ab.md`).

Two things fall out, and neither was visible from inside either issue alone.

### 1. T3 changed submission granularity 32× *mid-investigation*

| event | commit | time |
|---|---|---|
| riir-ai Bench 649 measures contention divergence | `9bc552e2` | **09:16** |
| this issue's T3 sets `CUBECL_WGPU_MAX_TASKS=1024` default | `431c0856` | **10:20** |

So the headline 4.05% → 50.00% figures were measured at `tasks_max=32`
(~26 command-buffer submissions per forward). Everything after 10:20 runs at
**1024** — the entire forward in a *single* submission.

**Consequence: a re-run of riir-ai Bench 649 today is not comparing like with
like.** Anyone re-measuring that gate must pin `CUBECL_WGPU_MAX_TASKS`
explicitly and record which value they used, or the comparison silently spans a
32× change in how work reaches the driver. This is not hypothetical — Issue 640
has already had five experiments voided by uncontrolled variables.

### 2. `tasks_max` is an untested candidate *modulator* of the divergence

Submission granularity is exactly the kind of variable that could matter for a
cross-process interleaving effect: at 32 the victim yields to the driver ~26
times per forward, at 1024 once. Fewer, larger submissions mean fewer points at
which another process's work can interleave — or longer uninterrupted
occupancy. Either direction is plausible; neither has been measured.

This is a **cheap** test compared with everything else tried on Issue 640: it is
an env var, not a code change, and it slots directly into the existing
contention A/B as a third arm (`32` vs `1024` under identical contention).
Worth noting that six rounds of *building* a synthetic reproducer have already
failed there, so a free knob is disproportionately attractive.

**Recorded as a candidate, not a claim.** No measurement has been taken — the
GPU has been held by sibling jobs, and a 6%-scale or divergence-rate measurement
taken under contention would be worthless.

### 3. This issue's own measurement was taken correctly

The `tasks_max` sweep above explicitly states *"clean GPU state (no sibling
processes)"*. That was the right discipline and is worth affirming rather than
assuming: a 6% throughput delta measured while another process held the GPU
would have been indistinguishable from noise, given contention costs 1.78× on
its own. The result stands.

## Non-goals

- **CUDA path** — the cudarc path already uses CommandEncoder batching + dp4a at
  88.9% roofline (Issue 608). Not applicable.
- **F1 (RMSNorm folded into GEMV)** — non-goal per Issue 642 (doesn't compose
  with bit-plane dequant loop).
- **CubeCL graph capture** — CubeCL 0.11 exposes `start_capture`/`stop_capture`
  for CUDA-Graphs-style replay. Not applicable to decode (the `pos` parameter
  changes every token, invalidating the captured graph). May help prefill (fixed
  batch) — tracked separately if a prefill consumer materializes.
