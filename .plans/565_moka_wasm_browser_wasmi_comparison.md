# Plan 565: Moka WASM Port — Real Browser Side-by-Side + wasmi Comparison

**Date:** 2026-07-29
**Status:** ✅ COMPLETE (2026-07-30). New crate `crates/katgpt-moka-wasm` (dependency-free port, no `katgpt-core`). Real Moka JS built from source and measured in real Chrome via Playwright: **6.4 ms/move**, 140,850 B. Our WASM build measured the same way: without `simd128`, **8.6 ms — slower than Moka**; with `RUSTFLAGS='-C target-feature=+simd128'`, **0.6 ms — 10.7× faster than Moka**, at 269,405 B (1.9× larger, confirmed as predicted). wasmi (pure interpreter) on the same binary: 212 ms — 25× slower than JIT, confirming V8's JIT is where nearly all the performance lives. Full results in `.docs/06_game_arenas/go_arena.md` ("Plan 565" section).
**Depends on:** Plan 563 (native Rust Moka port + `GoMokaSearchPlayer`, complete), Issue 564 (ANE, closed negative)

## The gap this plan closes

Every "Moka is ~0.5 ms/move" number reported so far came from **our own native Rust port**, used as the baseline inside our own benchmark. We have never measured the actual browser-deployed Moka (million.dev/moka) — real JS↔WASM call overhead, engine dispatch, page/GC context. It's entirely possible plain `MokaPlayer` (zero search, one forward pass) is already faster than the real thing, purely because we skip the browser runtime entirely. This plan measures that instead of assuming it.

Two deliverables:
1. **Real, automated, side-by-side comparison** in an actual browser — our WASM build vs the real live Moka — for both bundle size and per-move latency, no self-benchmarking.
2. **wasmi** (pure-Rust WASM interpreter, no JIT) as a third data point on the same compiled `.wasm` binary, using this repo's own existing fuel-limited idiom (`src/pruners/bomber/wasm_pruner.rs`).

## What "win" means going in (stated honestly before measuring)

Three separate claims, not to be conflated:
- **Strength**: already settled (Plan 563) — `GoMokaSearchPlayer` beats Moka greedy 70.0% (n=300). Unaffected by anything in this plan.
- **Size**: our *weights* are identical to Moka's (it's their network). The only size question this plan can actually answer is: does our compiled artifact (wasm binary + glue JS + embedded weights) end up smaller or larger than Moka's real 134 KB (128 KB weights + 6 KB runtime, per their own disclosure — to be independently verified from real `Content-Length` headers, not trusted at face value)? Expect ours to be **larger** — we carry the full `GoState` rules engine, a JSON manifest parser, and (for the search config) an alpha-beta tree, none of which Moka's greedy-only runtime needs. Report whatever the real number is.
- **Speed**: this is the one actually in question. Native-vs-native, we already know our search config is ~5.3× slower per move than raw Moka (Plan 563). What's unmeasured: real-browser Moka vs our WASM build, for both "just the network" (comparable to raw Moka) and "network + search" (our winning config) configurations.

## Phase 0 — Recon: find Moka's real programmatic move API

Before scripting anything, read the cloned `github.com/millionco/moka` source (`/tmp/moka-repo` from Plan 563, or re-clone) for its actual exported interface — `src/client.ts`, `src/index.ts`, `src/worker.ts`. `tests/model-smoke.mjs` already showed `GoModelWorkerClient`, `encodeStudentFeatures`, `playMove` as real exports. Confirm whether the live site exposes these on `window` (or via a bundled script we can call into from Playwright's `page.evaluate`), so latency can be measured around the actual inference call rather than simulated mouse clicks on a canvas (clicks would add UI/animation noise on top of the number we actually want).

- **T0.1**: Identify the real inference entry point and its call signature.
- **T0.2**: Confirm whether `million.dev/moka` ships the same code as the GitHub repo (version drift was already found once in Plan 563 — the live `.bin.gz` didn't match the repo's `go-model.bin` byte-for-byte). If it's diverged again, note it; don't assume.

## Phase 1 — Real live-Moka baseline (the number we don't have yet)

- **T1.1**: Playwright script (Node, already available on this machine) opens `https://million.dev/moka` headless, waits for the model to load, and calls the real inference entry point from T0 directly (not via simulated clicks) for N=50+ distinct board positions.
- **T1.2**: Wrap each call in `performance.now()` inside the page context (not measured from the Node side, to exclude Playwright/CDP round-trip overhead from the number). Discard the first few calls as warmup, exactly like every CPU benchmark this session did.
- **T1.3**: Capture real transferred byte sizes from Playwright's response/network events for the actual weights + JS + WASM files served — an independently-measured bundle size, not a re-quote of their README's "134 KB" claim.
- **Output**: real `moka_browser_latency_us` (median + distribution, not just one sample) and real `moka_browser_bundle_bytes`.

## Phase 2 — Compile our Moka port to WASM

- **T2.1**: `wasm32-unknown-unknown` target (already installed) + `wasm-bindgen` (already installed) for `crates/katgpt-pruners`'s `moka_net` module. Two separate build configs, each a fair comparison to a different Moka baseline:
  - **(a) Network-only**: `MokaWeights::load()` + `forward()` + argmax — the direct analogue to Moka's own greedy runtime. This is the fair size/speed comparison.
  - **(b) Network + search**: adds `GoMokaSearchPlayer` (alpha-beta, depth=1, top_k=4) — our actual winning config, reported as its own row, not conflated with (a).
- **T2.2**: `wasm-opt -O3` (already installed) on both, matching the level of optimization Moka's own build presumably applies (verify from their `vite.config.ts`/build scripts what they actually do, rather than assume `-O3` is the fair comparison point).
- **T2.3**: Record real `.wasm` size + `wasm-bindgen`-generated JS glue size for both configs. This is where the "likely larger than Moka" prediction gets either confirmed or falsified — report whichever it is.
- **T2.4**: Minimal HTML/JS harness exposing one `make_move(board_state) -> (move, latency)`-shaped call per config, mirroring Moka's own interface shape from Phase 0 so the Phase 3 driver script can treat both sides uniformly.

## Phase 3 — Real side-by-side, same machine, same browser

- **T3.1**: One Playwright script loads our harness page (T2.4) and the real Moka page (T1) in the same browser session, runs the identical N=50+ positions through both, same warmup discipline as T1.2.
- **T3.2**: Single output table: `{real Moka, our network-only, our network+search} × {bundle bytes, median latency us, p50/p99}`. Every cell either measured directly or explicitly marked "not measured."
- **T3.3**: Sanity check against the Plan 563 native numbers (network-only forward pass was ~0.45–0.54 ms natively) — if the WASM-in-browser number for the *same config* comes out wildly different, that's a signal to investigate (WASM overhead, `wasm-bindgen` marshalling cost) rather than silently report it.

## Phase 4 — wasmi (pure interpreter, no JIT) on the identical binary

- **T4.1**: Take the exact `.wasm` binary from T2.1(a) (network-only) — same artifact, not a separate build — and run it through `wasmi`, mirroring `wasm_pruner.rs`'s fuel-limited `Engine`/`Store`/`Linker`/`TypedFunc` pattern already established in this repo.
- **T4.2**: Time N=50+ calls natively (no browser, no Playwright) — this isolates "interpreted WASM" cost from "browser JS engine" cost, giving a 4-way ladder: native Rust → wasmi-interpreted wasm → browser-JIT wasm (ours) → browser-JIT wasm (real Moka).
- **T4.3**: Fuel budget must be sized generously enough not to artificially truncate a real forward pass (~5.8M MACs) — start from `wasm_pruner.rs`'s existing `FUEL_PER_CALL` constant as a reference point, not a copy (that constant was sized for bomber's tiny validator, not a conv net).

## Phase 5 — Write-up

- **T5.1**: One results table in `.docs/06_game_arenas/go_arena.md`, all four rungs of the ladder, each number labeled with exactly how it was measured (native/wasmi/browser-ours/browser-real) so nothing gets silently re-quoted out of context later (the failure mode this whole plan exists to avoid).
- **T5.2**: Explicit final verdict sentence answering the user's actual question: is our WASM-in-browser build faster than real Moka, for the network-only config specifically (the fair comparison), separate from the network+search config (which is expected to be slower per-move, same as the native result, but hopefully still faster than real browser Moka's own latency if T2's "bring it home" hypothesis holds).

## Risks

- **Moka's real API might not be cleanly callable headlessly** — if `window`-level access isn't available (e.g. it's all inside a Web Worker with `postMessage`, per the README's mention of `GoModelWorkerClient`), Phase 0 needs to find the actual message-passing contract, not assume a direct function call.
- **Version drift** (already seen once) between the live site and the GitHub repo — if the live weights differ from what we vendored in Plan 563, the correctness-equivalence work doesn't automatically transfer; note but don't block on it, since Phase 1–3 only need Moka's *latency/size*, not bit-exact output matching.
- **`wasm-bindgen` glue size is a real cost** most naive comparisons forget — must be included in "our bundle size," not just the raw `.wasm` file, or the size comparison is dishonest in our favor.
- **Our wasm binary is very likely to be larger than Moka's** (full GoState engine + JSON parsing + search tree vs their minimal greedy runtime) — this is a probable, not merely possible, negative result on size. State it plainly if confirmed rather than reframing around whichever number looks best.
