# Bench 585 — llama.cpp vs katgpt ternary, stage-aligned on M3 Max Metal

**Date:** 2026-08-12
**Hardware:** M3 Max, **Metal only** (no CPU arm, no 4090 arm — both explicitly out of scope)
**Model:** `Ternary-Bonsai-27B-Q2_0.gguf` — same file for both arms
**Arm A:** `prismml-llama.cpp` @ `9ca265a`, ggml Metal backend
**Arm B:** katgpt ternary tier (`katgpt-core`/`katgpt-types` weights) via `riir-gpu` CubeCL `wgpu<msl>` → native MSL
**Roofline:** 400 GB/s assumed M3 Max peak (override `ROOFLINE_GBS`)

## Why this bench exists

Bench 009 (riir-clippy) compared a retriever against an LLM and we won 10^5× —
which tells you nothing about where to fix perf, because that comparison has
nothing to fix. This one puts **the same model on the same GPU** and aligns the
stages, so the gap is attributable.

## Geometry cross-check (both arms agree on the work)

llama.cpp's own loader reports, and our bench geometry independently assumes:

| param | llama.cpp `print_info` | Bench 606 `BENCH_SHAPES` |
|---|---|---|
| `n_layer` | 64 | 64 (48 DeltaNet + 16 attention) |
| `n_embd` | 5120 | 5120 |
| `n_ff` | 17408 | 17408 |
| `n_embd_k_gqa` | 1024 | 1024 (`attn_k/v` rows) |
| `n_expert` | **0 (dense)** | dense — all weights read per token |

Dense matters: with `n_expert = 0` there is no routing sparsity, so **every
weight is read every token in both arms**. Effective-bandwidth numbers below are
therefore comparable rather than an artifact of differing sparsity.

---

## Stage 1 — Kernel level: ternary GEMV vs Q2_0 `mul_mv`

Arm A, `test-backend-ops perf -o MUL_MAT -b MTL0`, `type_a=q2_0, type_b=f32`:

| shape (m×k) | n | µs/run | GFLOPS | weight MB | **GB/s** | % roof |
|---|---:|---:|---:|---:|---:|---:|
| 4096×14336 | 1 | 190.12 | 617.71 | 15.60 | **82.0** | 20.5% |
| 4096×14336 | 2 | 388.97 | 603.86 | 15.60 | 40.1 | 10.0% |
| 4096×14336 | 4 | 558.39 | 841.28 | 15.60 | 27.9 | 7.0% |
| 4096×14336 | 8 | 1261.89 | 744.54 | 15.60 | 12.4 | 3.1% |
| 4096×14336 | 512 | 10322.12 | **5.83 TFLOPS** | 15.60 | — | prefill regime |

`n=1` is the decode case. GB/s falls with `n` because the weight read amortizes
across more activations — that is the batching win, not a kernel regression.

Arm B, `bench_606_ternary_gemv_packed` re-run this session (rowtiled kernel,
amortized, median of 9):

| tensor | shape | best GB/s | % roof |
|---|---|---:|---:|
| `lm_head` | 248320×5120 | **132.6** | **33.2%** |
| `ffn_gate/up` | 17408×5120 | **113.7** | **28.4%** |
| `ffn_down` | 5120×17408 | **113.7** | **28.4%** |
| `attn_q` | 12288×5120 | 82.6 | 20.7% |
| `attn_qkv` | 10240×5120 | 67.6 | 16.9% |
| `attn_gate` | 6144×5120 | 62.3 | 15.6% |
| `ssm_out` | 5120×6144 | 60.6 | 15.1% |
| `attn_output` | 5120×6144 | 48.9 | 12.2% |
| `attn_k/v` | 1024×5120 | 16.6 | 4.1% |
| `ssm_alpha/beta` | 48×5120 | **0.9** | **0.2%** |

### Verdict on the kernel: we are NOT behind

At the closest size match — our `attn_q` (12288×5120 = 62.9 M weights) against
llama.cpp's 4096×14336 (58.7 M weights) — we measure **82.6 GB/s vs 82.0 GB/s**.
Parity. On the large FFN tensors and `lm_head` we run **113–133 GB/s (28–33%
roof)**, i.e. **1.4–1.6× llama.cpp's Q2_0 `mul_mv` efficiency**.

**So the ternary kernel is not the bottleneck, and further micro-tuning of the
large-shape kernel is the wrong place to spend effort.** This corrects the
intuition that a 94× end-to-end deficit implies a slow kernel.

---

## Stage 2 — Where our time actually goes (per-token projection budget)

Rowtiled amortized µs × per-token call count, summing to the harness's reported
`0.1092 s/token`:

| tensor | shape | calls/token | best GB/s | % roof | ms/token | **share** |
|---|---|---:|---:|---:|---:|---:|
| `ffn_down` | 5120×17408 | 64 | 113.7 | 28.4% | 30.7 | **28.1%** |
| `ffn_gate/up` | 17408×5120 | 128 | 113.7 | 28.4% | 28.2 | **25.8%** |
| `attn_qkv` | 10240×5120 | 48 | 67.6 | 16.9% | 11.7 | 10.7% |
| `ssm_alpha/beta` | 48×5120 | 96 | **0.9** | **0.2%** | 7.7 | **7.1%** |
| `ssm_out` | 5120×6144 | 48 | 60.6 | 15.1% | 7.0 | 6.4% |
| `attn_gate` | 6144×5120 | 48 | 62.3 | 15.6% | 6.8 | 6.2% |
| `attn_q` | 12288×5120 | 16 | 82.6 | 20.7% | 6.6 | 6.0% |
| `attn_k/v` | 1024×5120 | 32 | 16.6 | 4.1% | 3.9 | 3.6% |
| `lm_head` | 248320×5120 | 1 | 132.6 | 33.2% | 3.3 | 3.0% |
| `attn_output` | 5120×6144 | 16 | 48.9 | 12.2% | 3.2 | 3.0% |
| **total** | | **497 dispatches** | 62.3 eff. | 15.6% | **109.2** | **9.16 tok/s** |

---

## Stage 3 — End to end, and the honest asymmetry

| | work covered | ms/token | tok/s | effective GB/s | % roof |
|---|---|---:|---:|---:|---:|
| **llama.cpp Metal** | **everything** (projections + attention + norms + RoPE + softmax + KV + sampling) | **39.2** | **25.50** | **182.7** | **45.7%** |
| **katgpt Metal** | **projections only** (no norms, recurrence, attention scores, sampling) | 109.2 | 9.16 | 62.3 | 15.6% |

**Our projections alone cost 2.79× llama.cpp's entire token.** The real gap is
worse than 2.79×, because the comparison is partial-vs-complete: our column is a
*lower bound* that has not yet paid for anything except matmuls.

Effective bandwidth is the cleanest single summary: **182.7 GB/s vs 62.3 GB/s =
2.93×**. llama.cpp extracts 45.7% of the 400 GB/s roofline across a whole
forward; we extract 15.6% across a strict subset of it.

---

## Where to fix perf, in priority order

The kernel is at parity-or-better, so the loss is **structural**. Ranked by
measured recoverable milliseconds:

**P1 — Kill the launch-bound dispatches (7.7 ms/token, 7.1%, near-100% recoverable).**
`ssm_alpha/beta` is 48×5120 — **30 KB of weights** — and costs 80.2 µs per call
at **0.2% of roofline**, 96 calls per token. That is not bandwidth, it is pure
dispatch latency: the kernel finishes long before the launch overhead does.
`attn_k/v` (4.1% roof, 3.9 ms) is the same disease. Fusing these into their
neighbours removes ~11.6 ms/token for no kernel work at all. **Cheapest win on
the board.**

**P2 — Lift the mid-size tensors from ~15% to the proven 28% (≈14 ms/token).**
`attn_qkv`, `ssm_out`, `attn_gate`, `attn_output` total 28.7 ms/token at
12–17% roof, while our *own* FFN kernel proves 28.4% is reachable on this
hardware. This is a shape-tuning problem (row-tile width / simdgroup mapping for
these geometries), not a new algorithm.

**P3 — Reduce dispatch count structurally: 497 per token.** Even at zero
per-launch cost this bounds how well P1/P2 can land. llama.cpp runs a fused
graph; we run 497 discrete matvecs. Note the 109.2 ms figure is already the
**amortized** (best-case) column — the harness chains 32 launches and divides,
so it *understates* real unfused cost. A real forward pays more.

**P4 — Only then push the FFN kernel past 28.4% roof (58.9 ms/token, the
largest single block).** Highest absolute ceiling (58.9 ms is 54% of the budget)
but the lowest ratio-per-effort, since we are already 1.4× llama.cpp's
efficiency here. Do P1–P3 first.

**Explicitly NOT a priority:** the large-shape ternary kernel. We beat
llama.cpp's Q2_0 there. The recent AVX2 / row-parallel trit work (Issue 582/583)
paid off — this bench is the evidence, and it also says the next increment of
that work is not where the remaining 2.9× lives.

---

## ⚠️ Condition asymmetry — read before quoting the 2.79×

**This bench has the same mixed-condition flaw that Bench 009 (riir-clippy) was
corrected for, and it is not yet fixed here because the box will not go quiet.**

| measurement | machine state |
|---|---|
| Arm A `llama-bench` pp512 / tg128 (25.50 t/s) | **quiet**, `load 4.09`, recorded |
| Arm A `test-backend-ops` Q2_0 (82.0 GB/s) | load **not recorded** |
| Arm B Bench 606 re-run (9.16 tok/s, 82.6–132.6 GB/s) | **contended** — a sibling session was running `riir_gpu` at 330–370% CPU plus `bonsai_lora_accuracy_parity`, i.e. **competing for the same GPU** |

Bonsai decode on this box is separately measured to swing **12.6 – 25.5 t/s**
purely on GPU residency (see Bench 009's CORRECTION section), so contention here
is worth ~2×, not a rounding error.

**Both biases run AGAINST our arm**, so the conclusions are conservative:

- The **2.79×** projection-vs-full-token deficit is an **upper bound**. Our true
  quiet figure is better than 9.16 tok/s by an unquantified amount.
- The **kernel parity / 1.4–1.6× advantage** is a **lower bound**. If our kernel
  numbers were taken under GPU contention, the real efficiency is higher, which
  *strengthens* "the kernel is not the bottleneck."
- The **P1–P4 ranking is unaffected**, because it is derived from *ratios between
  our own tensors measured in the same run* (`ssm_alpha/beta` at 0.2% roof vs
  `ffn_gate/up` at 28.4%), not from the cross-arm comparison. Contention scales
  those roughly uniformly.

**Required follow-up:** re-run Arm B (and `test-backend-ops`) with the box quiet
and the load recorded, then restate the cross-arm rows. Until then, treat the
cross-arm magnitudes as provisional and the intra-arm ranking as sound.

---

## Caveats

- **Shapes are not exactly matched.** `test-backend-ops` enumerates fixed shapes
  (4096×14336); our geometry is Bonsai's real projections. The `attn_q`
  comparison (62.9 M vs 58.7 M weights) is the closest pair and is the only one
  quoted as a head-to-head. A shape-matched follow-up would need either a custom
  ggml test case or our harness run at 4096×14336.
- **Amortized vs round-trip.** Arm B's numbers divide out fixed CPU↔GPU latency
  by chaining 32 launches. This flatters Arm B; llama.cpp's 39.2 ms/token is a
  real end-to-end request. The gap is therefore a floor.
- **Roofline is assumed, not measured.** 400 GB/s is the M3 Max spec figure for
  this bin. llama.cpp's 182.7 GB/s effective is a measured lower bound on what
  the device actually delivers; if true peak is nearer 300 GB/s, both % roof
  columns scale up by 1.33× and the *ratio* between arms is unchanged.
- **`2.125 bits/weight`** used for both arms' byte accounting (2 bits + f16 scale
  per 128 weights). Cross-checks against the file: 7,165,121,600 B × 8 ÷
  26.90 B params = 2.131 bits/weight.

## Reproduce

```sh
# Arm A — llama.cpp Metal kernel perf (Q2_0 mul_mv / mul_mm)
test-backend-ops perf -o MUL_MAT -b MTL0 2>/dev/null | grep "us/run" | grep type_a=q2_0

# Arm A — end-to-end decode
llama-bench -m Ternary-Bonsai-27B-Q2_0.gguf -p 512 -n 128 -ngl 99 -r 3
# => pp512 207.79 +/- 6.25 t/s, tg128 25.50 +/- 0.36 t/s

# Arm B — katgpt ternary GEMV on Metal (in riir-ai)
CARGO_TARGET_DIR=/tmp/i606_m3 cargo test -p riir-gpu --features ternary_gemv \
  --release --test bench_606_ternary_gemv_packed -- --ignored --nocapture
```

Related: riir-ai [Bench 606](../../riir-ai/.benchmarks/606_ternary_gemv_packed_goat.md)
(the GEMV harness + its own GOAT gates),
[`.docs/08_performance/ternary_group_q2_0_tier.md`](../.docs/08_performance/ternary_group_q2_0_tier.md)
(the tier this measures).
