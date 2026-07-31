// bench_puct.js — PUCT latency under V8 JIT (Node.js), same engine as Chrome.
//
// Loads the wasm-bindgen nodejs output + times the arena C-ABI exports.
// Build the wasm first:
//   ./scripts/build-moka-wasm.sh            # outputs to crates/katgpt-moka-wasm/target-wasm/nodejs/
//   node crates/katgpt-moka-wasm/bench/bench_puct.js
//
// What this measures: median ms/move for PUCT search at a given budget + batch_k.
// The `setupFixture` cost (init + 8 forced opening moves) is measured separately
// and subtracted, so the reported number is pure search time.
//
// The fixture (8 opening moves at indices 40,41,31,50,32,49,22,58) produces a
// mid-game 9×9 position with both players having played — the regime where
// PUCT's policy prior actually matters (early-game positions are too uniform).

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

// c_puct is passed as a bit-reinterpreted f32 (the C-ABI takes u32 bits).
const buf = new ArrayBuffer(4);
new Float32Array(buf)[0] = 1.5;  // c_puct = 1.5
const cPuctBits = new Uint32Array(buf)[0];

function setupFixture(budget, top_k, batch_k) {
  ex.wasmi_arena_init(budget, cPuctBits, top_k, batch_k);
  // 8 forced opening moves — same fixture as the Issue 204/205 measurements.
  for (const idx of [40, 41, 31, 50, 32, 49, 22, 58]) {
    ex.wasmi_arena_play(idx);
  }
}

function bench(budget, top_k, batch_k, label) {
  // Warm up (first call JIT-compiles; subsequent calls run compiled).
  setupFixture(budget, top_k, batch_k);
  ex.wasmi_arena_search_puct();

  const N = 10;
  // Total time: setup + search, N iterations.
  const t0 = process.hrtime.bigint();
  for (let i = 0; i < N; i++) {
    setupFixture(budget, top_k, batch_k);
    ex.wasmi_arena_search_puct();
  }
  const with_setup_ms = Number(process.hrtime.bigint() - t0) / 1e6 / N;

  // Setup-only time, to subtract.
  const t1 = process.hrtime.bigint();
  for (let i = 0; i < N; i++) {
    setupFixture(budget, top_k, batch_k);
  }
  const setup_ms = Number(process.hrtime.bigint() - t1) / 1e6 / N;

  const per_move_ms = with_setup_ms - setup_ms;
  console.log('  ' + label.padEnd(20) + ': ' + per_move_ms.toFixed(1) + ' ms/move');
  return per_move_ms;
}

// ── Sequential (batch_k=1, the parity path — bit-identical to wasmi) ────
console.log('=== PUCT latency (Node.js V8 JIT — setup subtracted) ===');
console.log('');
console.log('Sequential (batch_k=1):');
const seq_b50  = bench(50,  8, 1, 'PUCT b50 K=1');
const seq_b100 = bench(100, 8, 1, 'PUCT b100 K=1');
const seq_b200 = bench(200, 8, 1, 'PUCT b200 K=1');

// ── Batched (batch_k=8, Issue 205 opt-in path) ──────────────────────────
console.log('');
console.log('Batched (batch_k=8, Issue 205):');
const bat_b50  = bench(50,  8, 8, 'PUCT b50 K=8');
const bat_b100 = bench(100, 8, 8, 'PUCT b100 K=8');
const bat_b200 = bench(200, 8, 8, 'PUCT b200 K=8');

console.log('');
console.log('=== Speedup (sequential / batched) ===');
console.log('  b50:  ' + (seq_b50  / bat_b50 ).toFixed(2) + 'x  (' + seq_b50 .toFixed(1) + ' -> ' + bat_b50 .toFixed(1) + ' ms)');
console.log('  b100: ' + (seq_b100 / bat_b100).toFixed(2) + 'x  (' + seq_b100.toFixed(1) + ' -> ' + bat_b100.toFixed(1) + ' ms)');
console.log('  b200: ' + (seq_b200 / bat_b200).toFixed(2) + 'x  (' + seq_b200.toFixed(1) + ' -> ' + bat_b200.toFixed(1) + ' ms)');

console.log('');
console.log('=== Sanity floor ===');
console.log('  If seq_b50 is ~500ms instead of ~30ms, the SIMD flag was lost.');
console.log('  Rebuild with: ./scripts/build-moka-wasm.sh');
