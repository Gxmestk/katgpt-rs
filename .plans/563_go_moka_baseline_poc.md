# Plan 563: Go Modelless-vs-Moka Baseline (PoC)

**Date:** 2026-07-29 (revised same day — pivoted from proxy to real weights)
**Status:** ✅ COMPLETE (2026-07-29), then EXTENDED same day. All planned phases executed: real Moka v1 weights vendored + ported natively to Rust (`crates/katgpt-pruners/src/go/moka_net.rs`), parity-checked (Rust-only, no Python/Node), wired into a head-to-head tournament (`examples/go_11_moka_head_to_head.rs`). GNU-Go-proxy appendix never needed — parity checks passed first try.

**Baseline verdict (the plan's actual question):** **Moka wins 100% vs every modelless player** — 0/20 for each of Greedy/Validator/HL/GZero/MCTS, by 35–80 point margins on an 81-point board. The answer to "can our modelless architecture beat Moka" is **no**.

**Post-plan extension (beyond original scope):** two follow-ups asked whether Moka could be *beaten at all* by composing with it rather than competing against it.
- `MokaHeuristic` / `GoMctsMokaPlayer` — Moka's **value** head as an MCTS leaf evaluator: **0/10 at two configs** (random rollout, and immediate in-distribution eval). Diagnosis: plain UCB1 at budget ≈ branching factor gives ~1 visit/arm, and the **policy** head went unused.
- `GoMokaSearchPlayer` — Moka's **policy** head prunes (top-K beam) + **value** head evaluates leaves + alpha-beta negamax: **210W/90L = 70.0%** over 300 independent games at depth=1, top_k=4 (exact 1-sided p = 1.7×10⁻¹², 95% CI [64.8%, 75.2%]) — a decisive edge. **Not a modelless win**: it requires Moka's weights to function. Framing: "search on top of their model beats their model."
  - **Key tuning finding: narrow beam beats wide, on both strength and latency.** top_k 3–6 gives ~68–74%; top_k 8–16 gives ~58–60% (k=4 vs k=8: z = 2.42, p ≈ 0.016). Moka's policy head is more reliable than its value head, so a wide beam just gives value-head noise more chances to promote a blunder. This also retro-explains the depth-2 null result: more of the same noisy evaluator doesn't help, constraining its scope does.
  - Earlier reported figures were all mistuned or undersampled: 63.3% (n=30), 61.7% (n=60), **57.0% (n=200 — honest sampling but top_k=8, a bad config)**, 74.0% (n=100, small-sample high). Quote only the n=300 figure.

**Latency rework (follow-up, prompted by 119 ms/move being unacceptable):** **119 ms → ~2.8 ms per move (~42×)**, from three fixes:
- (a) The forward-pass kernel was itself the bottleneck at only ~1.7 GMAC/s. Loop reorder so position is outermost (gather each `k*k*in_ch` patch once, reuse across all output channels) + 1×1-conv fast path + delegating the dot product to **`katgpt_types::simd::simd_dot_f32`** ⇒ **8.7× total**, 3.4 ms → ~0.4–0.45 ms/forward (~13–15 GMAC/s).
- (b) depth=2 measured *identical* strength to depth=1 at 4× the cost ⇒ depth 1.
- (c) top_k 8 → 4 ⇒ ~2× cheaper *and* significantly stronger (see above).

Correctness pinned by keeping the naive conv/linear as an equivalence oracle at every layer shape the net uses, plus a bit-identity test on scratch reuse (the SIMD path changes summation order *and* uses FMA single-rounding, so this is not optional).

**Two honest misses worth remembering:**
1. **DRY:** the first pass hand-rolled 8 accumulator chains hoping LLVM would vectorize them. `katgpt_types::simd::simd_dot_f32` — already a dependency, already used elsewhere in this same crate — is explicit NEON/AVX2-FMA intrinsics and was **1.36× faster**. Reuse first, hand-roll never.
2. **The allocation hypothesis was wrong.** Removing ~50 `vec![]`/forward measured −1.7% in one run and +22.7% in another — inside the noise floor. All real gain came from cache locality and vectorization. Scratch plumbing retained for timing stability only.

**Also checked and rejected as non-applicable** (asked whether AHLA/LEO/MUX/plasma-tier were relevant): HLA/AHLA is a Transformer *attention-layer* replacement — Moka is a pure CNN with no attention or sequence dimension; LEO is flow-field/QGF navigation and goal networks; MUX is expert routing, which cannot manufacture strength from ingredients that individually score 0%. Plasma-tier is half-relevant: its `BinaryPlasma`/`PlasmaPath` kernels are 1–2 bit **matvec** (not conv) and would likely wreck a 105K-param int8 net's accuracy, but the direction it points at — lower-precision SIMD — is the genuine remaining lever (int8 inference; Moka's weights are already int8 and are currently dequantized to f32 at load).

Two measurement bugs were found and fixed during the extension (per-game history leakage; deterministic players making "N games" ≈ 2 distinct samples until randomized openings were added) — see `.docs/06_game_arenas/go_arena.md` for both, and disregard any win-rate reported over deterministic repeats.
**Depends on:** `crates/katgpt-pruners/src/go/*` (this repo), `riir-ai/.plans/393_go_gemma_arena.md` + `408` + `410` (sibling repo, already complete, cited not re-run)

## Revision note

Original draft of this plan used GNU Go as a strength proxy because Moka appeared to be browser-only. That's wrong — Moka ships its actual weights as a direct-download, MIT-licensed, fully open-source artifact:

- Live download: `https://million.dev/models/moka-model-4a58dcfd.bin.gz` (confirmed reachable, 128,100 bytes gzipped → 138,616 bytes decompressed)
- Source + reference build checked into GitHub: `github.com/millionco/moka` (MIT license), including a **committed copy of the shipped model** — `model/go-model.bin` (113,648 bytes, byte-exact match to its own manifest's `weightsBytes`) + `model/go-model.json` (full tensor manifest: name → shape → dtype → byte offset → per-channel dequant scale offset) + the exact PyTorch/MLX-equivalent architecture in `python/go_model/model.py` + the exact input feature encoding in `python/go_model/features.py`.

This means we can do for Moka exactly what `riir-ai` Plan 393 did for Gemma 2 — load the real published weights and run a real forward pass, head-to-head against our players — except Moka is dramatically simpler than Gemma 2 (105,353 params, plain conv/linear, no attention/RoPE/tokenizer/autoregressive decode — one forward pass per move) so a from-scratch Rust port is realistic within this PoC's scope, not a multi-week undertaking. **Note:** the live site's `.bin.gz` (138,616 B decompressed) does not match the repo's `go-model.bin` (113,648 B) byte-for-byte — the live build has likely moved on since the repo snapshot. Use the repo's `go-model.bin` + `go-model.json` pair (they're mutually consistent and MIT-licensed) as the primary source; treat the live `.bin.gz` as a secondary check only if we want the exact currently-deployed weights (would need to re-derive/confirm its own manifest first — the JSON manifest isn't served alongside it as a separate file the way the repo ships one).

## Confirmed architecture (from `model/go-model.json` + `python/go_model/model.py`)

`MokaGlobalResidualNetwork` — 9×9 board only, `residualBlockKind: "nested-bottleneck"`, `globalResidualBlockInterval: 4`:

```
Input: 12 planes × 9×9 (see feature encoding below)
Stem:  Conv2d(12→32, k=3, pad=1) + ReLU                         [trunk channels = 32]
12× NestedBottleneckBlock (bottleneck channels = 16):
  reduce:  Conv2d(32→16, k=1) + ReLU
  first:   Conv2d(16→16, k=3, pad=1) + ReLU
  [every 4th block (index 3, 7, 11): GlobalNestedBottleneckBlock —
     after `first`, global_values = concat(mean_pool(hidden), max_pool(hidden))  # 32-d
     global_hidden = ReLU(Linear(32→8))
     global_bias   = Linear(8→16)             # zero-initialized at train start
     hidden = hidden + global_bias  (broadcast spatially)  before `second`]
  second:  Conv2d(16→16, k=3, pad=1) + ReLU
  expand:  Conv2d(16→32, k=1)
  output:  ReLU(input + expand_output)                          # residual add
Policy head: Conv2d(32→4, k=1) + ReLU → flatten(4×81=324) → Linear(324→82)
Value head:  Conv2d(32→2, k=1) + ReLU → flatten(2×81=162) → Linear(162→32) + ReLU
             → Linear(32→1) → tanh
```

Policy output: 82 logits = 81 board points (row-major, `divmod(move, 9)`) + 1 pass (index 81). Value output: scalar in [-1,1], current-player-perspective win estimate. Moka's own arena config (`OUTCOME_MOKA_SAMPLING_TEMPERATURE = 0.0`) plays it greedy (argmax policy over legal moves) — mirror that for the baseline, no search bolted on, to keep the comparison "network vs heuristics," not "network+search vs heuristics."

**Quantization** (`python/go_model/quantization.py`): symmetric per-output-channel int8. For each weight tensor, `scale[c] = max(abs(weight[c, ...])) / 127` (one scale per output channel — first dim); dequantized value = `int8_value * scale[c]`. Manifest confirms this layout: each `*.weight` entry has a `scaleOffset` immediately following its int8 data, sized `output_channels × 4 bytes` (float32), then the bias tensor (float32, no quantization) follows.

**Feature encoding** (`python/go_model/features.py::encode_moka_features`, 12 planes over the 9×9 grid):
| Plane | Meaning |
|---|---|
| 0 | current player's stones |
| 1 | opponent's stones |
| 2 | current-player stones in atari (group liberty count == 1) |
| 3 | opponent stones in atari |
| 4 | current-player stones with exactly 2 liberties |
| 5 | opponent stones with exactly 2 liberties |
| 6 | ko point |
| 7 | last move (1 ply ago) |
| 8 | move 2 plies ago |
| 9 | whole-plane fill if the move 1 ply ago was a pass |
| 10 | whole-plane fill if the move 2 plies ago was a pass |
| 11 | perspective komi, constant-filled: `(-7.0 * next_color) / 15.0` |

Board/color convention: `next_color` is ±1 (not 0/1); board cells are `0` (empty) / `±1` (color). Our `GoState`/`GoCell` (Black/White/Empty) needs a small adapter, not a rewrite — group/liberty detection already exists in `state.rs`/`GoHeuristic` per the existing docs, reuse it rather than reimplementing `get_group`.

## Goal

One reproducible table + the harness, with Moka's row now **measured, not cited**:

| Player | Params | Payload size | Win% vs Random | Win% vs Moka (real weights) | Latency/move | 
|---|---|---|---|---|---|
| Moka (this repo, real MIT weights) | 105,353 | 113,648 B (+ manifest) | — | — | measured, native Rust, µs–ms range |
| Gemma 2 2B (external, riir-ai Plan 393/408/410, cited not rerun) | 2.6B (+10.4M LoRA) | ~4.9 GB f16 | 33% (=random) | not run | ~50 s/move CPU |
| Greedy / Validator / HL / GZero / MCTS (this repo, measured) | 0 (Q-tables: tens of f32s) | <1 KB | 100% (Greedy/Validator/HL) | **TBD — this PoC** | 2–8 µs |

## Phase 1 — Vendor + parse the real weights

- **T1**: Vendor `model/go-model.bin` + `model/go-model.json` from `github.com/millionco/moka` (MIT license — copyright notice must be retained per license terms) into this repo, e.g. `crates/katgpt-pruners/assets/moka/`. Small (113,648 + ~15 KB manifest), no runtime download dependency.
- **T2**: New module `crates/katgpt-pruners/src/go/moka_net.rs`. Parse `go-model.json` (serde: tensor name → `{dataOffset, dtype, shape, scaleOffset?}`), read `go-model.bin`, dequantize every `*.weight` tensor once at load time (`int8 * scale[out_channel]` → `Vec<f32>`), load `*.bias` tensors as-is (already float32). One-time cost, ~114 KB — trivial.
- **T3**: Unit test: load the manifest + bin, assert tensor byte ranges match declared shapes/offsets exactly (catches any transcription error in the loader before it silently produces garbage).

## Phase 2 — Forward pass + feature encoding

- **T4**: Implement the forward pass exactly per the architecture table above: stem conv, 12 nested-bottleneck blocks (3 with the global-pooling branch at indices 3/7/11), policy head, value head. Plain nested loops over the 9×9 grid — no SIMD needed at this size, but this repo already has SIMD conv/matmul primitives (`katgpt-types/src/simd/`) if we want to reuse them for the 1×1/3×3 convs.
- **T5**: Implement `encode_moka_features()` per the table above, adapting from our `GoState`. Requires: current board, per-stone-group liberty count (reuse `state.rs` group/liberty logic), ko point, last two moves (including pass), current color as ±1, and the fixed komi constant (7.0, per `KOMI_POINTS`) normalized by 15.0 — **note this fixed komi differs from our own arena's converged self-play komi (Plan 091, ~42 on 9×9)**; use Moka's own convention for this comparison, don't substitute ours.
- **T6**: `MokaPlayer: GoPlayer` — forward pass → mask illegal moves → argmax over legal policy logits (ties broken deterministically) → fall back to pass only if no legal move scores above the pass logit, mirroring greedy (`temperature=0.0`) play.
- **T7 (parity check, do not skip — Rust-only, no Python/Node runtime dependency)**: A subtly wrong conv padding, channel order, or dequant-scale application won't crash — it'll just produce a systematically weaker/different player and silently invalidate the whole comparison. Validate without shelling out to `uv sync`/MLX (Python) or `node` (the TS/WASM runtime): the repo's `tests/*.py` and `tests/model-smoke.mjs` are read as **static text** for any literal hardcoded input/output arrays they contain (transcribe the numbers into a Rust `#[test]` fixture — we're reading data out of the file, not executing it), plus an independent **self-cross-check written entirely in Rust**: hand-derive the expected stem-layer output (and first `NestedBottleneckBlock`) for a couple of trivial boards (empty board; single stone) directly from the raw manifest bytes (`go-model.json` offsets + `go-model.bin` int8/scale/bias) in a second, independently-written code path inside the test — not by calling the loader/forward-pass module under test — so a transcription bug in the real implementation has an independent check to catch it. If literal fixtures exist in the repo's test files, prefer those as the primary oracle; the hand-derived Rust cross-check is the fallback/supplement, not Python.

## Phase 3 — Run the real head-to-head tournament

- **T8**: New example `examples/go_11_moka_head_to_head.rs` — Greedy, Validator, HL, GZero, MCTS(200/500/1000) each play N=20+ games vs `MokaPlayer` on 9×9 (matches Moka's own board size exactly — no size mismatch), alternating colors, komi=7.0 (Moka's convention, per T5). Reuse `tournament.rs` + `analytics.rs` for aggregation, same pattern as `go_02_tournament`.
- **T9**: Record real measured latency/move for both sides natively (no browser/WASM overhead on either side now) — genuine perf comparison, not the "µs heuristic vs browser-JS ms" apples-to-oranges framing from the original ask.
- **T10**: Print the full comparison table with the Moka row now measured locally; Gemma2 row stays as a cited external reference (already exhaustively benchmarked, not worth re-running per the "explicitly out of scope" note below).

## Phase 4 — Honest write-up

- **T11**: Add results to `.docs/06_game_arenas/go_arena.md` — architecture summary, parity-check result (T7), win-rate/latency/size table, and a direct verdict on the original question: does our best modelless player beat Moka's real weights on 9×9, and at what latency/size cost or gain.
- **T12 (optional, only if T7 parity fails and can't be resolved)**: fall back to the original GNU-Go-proxy plan (kept below) as a secondary real-strength anchor — GNU Go 3.8 is already confirmed installed locally (`/opt/homebrew/bin/gnugo`).

## Explicitly out of scope

- Re-running or extending the Gemma2 Go experiment — already exhaustively benchmarked in `riir-ai` (0% strength gain over random, every modelless technique + tiny LoRA tried). Cite, don't re-run.
- Any attempt to retrain, fine-tune, or improve Moka's weights, or to train a competing distilled CNN of our own — this PoC measures the existing artifact, it doesn't build a new one.
- Scraping the live `million.dev/moka` web widget — no longer needed now that the real weights are directly available MIT-licensed.

## Risks

- **Parity risk is the whole ballgame** (T7) — a CNN port with a subtly wrong padding, channel order, or dequant scale application will run without error and just be quietly wrong. Do not report any win-rate number from `MokaPlayer` until T7 passes.
- License compliance: MIT requires retaining the copyright notice — carry `LICENSE`/`THIRD_PARTY_NOTICES.md` attribution alongside the vendored files (T1).
- The live-site `.bin.gz` vs repo `go-model.bin` size mismatch (138,616 B vs 113,648 B) means "the exact model currently on million.dev" and "the model this PoC benchmarks" may not be bit-identical — call this out explicitly in the write-up (T11), don't imply we tested the live site.

---

## Appendix: original GNU-Go-proxy plan (superseded, kept as Phase 4 fallback)

GNU Go 3.8 is confirmed installed locally (`/opt/homebrew/bin/gnugo`). If the native Moka port stalls on parity (T7), fall back to:
- Minimal GTP subprocess client (`gnugo --mode gtp --level N`) implementing the existing `GoPlayer` trait.
- Tournament vs GNU Go level 10 as a real, locally-runnable, roughly-kyu-anchored opponent instead of a bit-exact Moka comparison.
- Explicit caveat: GNU Go level→kyu is informal, and GNU-Go-vs-KataGo is not an equivalent evaluation protocol to Moka's self-reported number — no cross-engine ELO claimed.
