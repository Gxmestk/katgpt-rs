// Fair side-by-side: real Moka JS (their actual code) vs our WASM, both in Node V8 JIT.
// Re-runs the Plan 565 measurement without needing Playwright/Chrome.
//
// Build prerequisites:
//   cd /tmp/moka-repo && pnpm install && pnpm run build
//   cd /Users/katopz/git/katgpt-rs && ./scripts/build-moka-wasm.sh
//
// Run:
//   node crates/katgpt-moka-wasm/bench/bench_moka_side_by_side.mjs

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);

// ── Load real Moka (ESM) ─────────────────────────────────────────
const MOKA_REPO = process.env.MOKA_REPO ?? "/tmp/moka-repo";
const moka = await import(path.join(MOKA_REPO, "dist/index.js"));

const manifest = JSON.parse(fs.readFileSync(`${MOKA_REPO}/model/go-model.json`, "utf8"));
const weights_bytes = fs.readFileSync(`${MOKA_REPO}/model/go-model.bin`);
const weightsBuffer = weights_bytes.buffer.slice(
  weights_bytes.byteOffset,
  weights_bytes.byteOffset + weights_bytes.byteLength,
);
const runtime = moka.GoModelRuntime.create(manifest, weightsBuffer);

// ── Load our WASM (CJS via createRequire) ───────────────────────
const GLUE = path.join(path.dirname(new URL(import.meta.url).pathname), "..", "target-wasm", "nodejs", "katgpt_moka_wasm.js");
const our = require(GLUE);

const game = new our.WasmGame();
const net = new our.WasmMoka();

// Same opening sequence as Moka's model-smoke test, for fair comparison.
const OPENING = [20, 24, 56, 60];
for (const idx of OPENING) game.play(idx);

// ── Build Moka's input features (their encoding) ────────────────────
let mokaGameState = moka.createGameState();
for (const move of OPENING) {
  mokaGameState = moka.playMove(mokaGameState, move);
}
const mokaFeatures = moka.encodeStudentFeatures(mokaGameState);

// ── Build our input features ────────────────────────────────────────
const ourFeatures = game.encode_features();
const ourFeaturesPtr = game.encode_features_ptr();

// ── Sanity: both produce finite output ──────────────────────────────
const mokaOut = runtime.infer(mokaFeatures);
const ourOut = net.infer(ourFeatures);
assert.ok(mokaOut.policyLogits.every(Number.isFinite), "Moka output finite");
assert.ok(mokaOut.policyLogits.length === 82, "Moka 82 logits");
assert.ok(Number.isFinite(ourOut[0]), "our output finite");

console.log(`Moka policyLogits.length = ${mokaOut.policyLogits.length}, value = ${mokaOut.value.toFixed(3)}`);
console.log(`Our output.length = ${ourOut.length} (82 logits + value)`);
console.log("");

// ── Bench: real Moka ────────────────────────────────────────────────
const N = 500;
// Warmup
for (let i = 0; i < 100; i++) runtime.infer(mokaFeatures);

let mokaSum = 0;
const t0 = performance.now();
for (let i = 0; i < N; i++) {
  const r = runtime.infer(mokaFeatures);
  mokaSum += r.value;
}
const t1 = performance.now();
const mokaMs = (t1 - t0) / N;

console.log(`Real Moka JS (their actual dist/index.js, V8 JIT) - ${N} iters`);
console.log(`  median per call: ${mokaMs.toFixed(3)} ms`);
console.log(`  (sum check: ${mokaSum.toFixed(1)})`);
console.log("");

// ── Bench: our WASM greedy (marshalled API) ─────────────────────────
for (let i = 0; i < 100; i++) net.infer(ourFeatures);

let ourSum = 0;
const s0 = performance.now();
for (let i = 0; i < N; i++) {
  const r = net.infer(ourFeatures);
  ourSum += r[0];
}
const s1 = performance.now();
const ourMarshalledMs = (s1 - s0) / N;

console.log(`Our WASM greedy (WasmMoka.infer, marshalled API, V8 JIT) - ${N} iters`);
console.log(`  median per call: ${ourMarshalledMs.toFixed(3)} ms`);
console.log("");

// ── Bench: our WASM greedy (zero-copy API) ──────────────────────────
for (let i = 0; i < 100; i++) net.infer_ptr(ourFeaturesPtr);

let our2Sum = 0;
const u0 = performance.now();
const mem = our.wasmMemory().buffer;
for (let i = 0; i < N; i++) {
  const optr = net.infer_ptr(ourFeaturesPtr);
  const view = new Float32Array(mem, optr, 1);
  our2Sum += view[0];
}
const u1 = performance.now();
const ourZeroCopyMs = (u1 - u0) / N;

console.log(`Our WASM greedy (WasmMoka.infer_ptr, zero-copy API, V8 JIT) - ${N} iters`);
console.log(`  median per call: ${ourZeroCopyMs.toFixed(3)} ms`);
console.log("");

// ── Verdict ────────────────────────────────────────────────────────
const speedup_marshalled = mokaMs / ourMarshalledMs;
const speedup_zerocopy   = mokaMs / ourZeroCopyMs;

console.log("=== Side-by-side verdict ===");
console.log(`Moka JS            : ${mokaMs.toFixed(3)} ms/call`);
console.log(`Ours (marshalled)  : ${ourMarshalledMs.toFixed(3)} ms/call  ->  ${speedup_marshalled.toFixed(2)}x ${speedup_marshalled >= 1 ? "faster" : "SLOWER"} than Moka`);
console.log(`Ours (zero-copy)   : ${ourZeroCopyMs.toFixed(3)} ms/call  ->  ${speedup_zerocopy.toFixed(2)}x ${speedup_zerocopy >= 1 ? "faster" : "SLOWER"} than Moka`);
console.log("");
console.log("Plan 565 documented: Moka 6.4 ms, ours 0.6 ms marshalled / 0.5 ms zero-copy = 10.7x / 12.8x");
