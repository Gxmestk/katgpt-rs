// bench_k_sweep.js — batched MCTS K-sweep (Issue 205).
//
// Measures ms/move at budget=50 for batch_k = 1,2,4,8,16,25,50.
// Used to produce the "diminishing returns" table in Benchmark 205:
//   K=1: 1.00×  K=8: 1.09×  K=50: 1.19×
//
// The honest finding: the forward pass is compute-bound (FPU-saturated SIMD
// dot kernel), not cache-bound. Batching K samples through the same weight
// slice doesn't reduce total FLOPs. See .benchmarks/205_puct_wasm_batched_mcts_latency.md.
//
// Build first:  ./scripts/build-moka-wasm.sh
// Run:          node crates/katgpt-moka-wasm/bench/bench_k_sweep.js

const path = require('path');
const GLUE = path.join(__dirname, '..', 'target-wasm', 'nodejs', 'katgpt_moka_wasm.js');

let moka, ex;
try {
  moka = require(GLUE);
  ex = moka.__raw_wasm;
} catch (e) {
  console.error('error: cannot load wasm glue at ' + GLUE);
  console.error('  did you run ./scripts/build-moka-wasm.sh first?');
  console.error('  (' + e.message + ')');
  process.exit(1);
}

const buf = new ArrayBuffer(4);
new Float32Array(buf)[0] = 1.5;
const cPuctBits = new Uint32Array(buf)[0];

function setupFixture(budget, top_k, batch_k) {
  ex.wasmi_arena_init(budget, cPuctBits, top_k, batch_k);
  for (const idx of [40, 41, 31, 50, 32, 49, 22, 58]) {
    ex.wasmi_arena_play(idx);
  }
}

function bench(budget, top_k, batch_k, label) {
  setupFixture(budget, top_k, batch_k);
  ex.wasmi_arena_search_puct();

  const N = 10;
  const t0 = process.hrtime.bigint();
  for (let i = 0; i < N; i++) {
    setupFixture(budget, top_k, batch_k);
    ex.wasmi_arena_search_puct();
  }
  const with_setup_ms = Number(process.hrtime.bigint() - t0) / 1e6 / N;

  const t1 = process.hrtime.bigint();
  for (let i = 0; i < N; i++) setupFixture(budget, top_k, batch_k);
  const setup_ms = Number(process.hrtime.bigint() - t1) / 1e6 / N;

  return with_setup_ms - setup_ms;
}

console.log('=== K sweep at budget=50 (Issue 205) ===');
const results = {};
for (const k of [1, 2, 4, 8, 16, 25, 50]) {
  results[k] = bench(50, 8, k, 'K=' + k);
  console.log('  K=' + String(k).padStart(2) + ': ' + results[k].toFixed(1) + ' ms/move');
}

console.log('');
console.log('Speedup vs K=1:');
for (const k of [2, 4, 8, 16, 25, 50]) {
  console.log('  K=' + String(k).padStart(2) + ': ' + (results[1] / results[k]).toFixed(2) + 'x');
}

console.log('');
console.log('Honest interpretation (from Benchmark 205):');
console.log('  The gain is marginal (1.09× at K=8) because the forward pass is');
console.log('  compute-bound, not cache-bound. K=1 stays the default (wasmi parity).');
