// bench_puct_int8.js — PUCT latency comparison: f32 vs int8 forward path (Issue 206).
//
// Measures median ms/move for PUCT search at b50/b100/b200 using both the
// f32 baseline (WasmPuctPlayer) and the int8 path (WasmPuctPlayerInt8).
//
// Build first:
//   ./scripts/build-moka-wasm.sh --nodejs
//   node crates/katgpt-moka-wasm/bench/bench_puct_int8.js
//
// Expected: int8 is faster on WASM (extmul workaround, ~2× on dominant conv
// sizes per Bench 565 T4). The forward pass is ~85% of PUCT latency.

const path = require('path');
const GLUE = path.join(__dirname, '..', 'target-wasm', 'nodejs', 'katgpt_moka_wasm.js');

let moka;
try {
  moka = require(GLUE);
} catch (e) {
  console.error('error: cannot load wasm glue at ' + GLUE);
  console.error('  did you run ./scripts/build-moka-wasm.sh first?');
  console.error('  (' + e.message + ')');
  process.exit(1);
}

// Fixture: 8 opening moves to reach a mid-game position where policy priors matter.
const OPENING_MOVES = [40, 41, 31, 50, 32, 49, 22, 58];

function makeBoard() {
  // The wasm-bindgen API takes (cells, to_play, ko_point, consecutive_passes).
  // We build a simple board state after the opening sequence.
  // For benchmarking, we use the board state AFTER the opening moves.
  // The arena API handles move replay internally, but the wasm-bindgen API
  // needs the final board state. Let's build it.
  const cells = new Uint8Array(81);
  // Simulate the opening on the JS side (simplified — alternating placements).
  let toPlay = 0; // 0=black
  for (const idx of OPENING_MOVES) {
    cells[idx] = toPlay + 1; // 1=black, 2=white
    toPlay = 1 - toPlay;
  }
  return { cells, toPlay, ko: 255, passes: 0 };
}

function benchPlayer(PlayerClass, budget, cPuct, topK) {
  const { cells, toPlay, ko, passes } = makeBoard();

  // Construct ONCE — weight loading happens here. The real production
  // pattern is construct-once + search-many (the player persists across
  // moves). Creating a new player per move would amortize weight loading
  // incorrectly.
  const player = new PlayerClass(budget, cPuct, topK);

  // Warmup (JIT compilation).
  player.search(cells, toPlay, ko, passes);

  const N = 10;
  const t0 = process.hrtime.bigint();
  for (let i = 0; i < N; i++) {
    player.search(cells, toPlay, ko, passes);
  }
  const ms = Number(process.hrtime.bigint() - t0) / 1e6 / N;
  player.free();
  return ms;
}

console.log('=== PUCT latency: f32 vs int8 (Node.js V8 JIT) ===');
console.log('');

const budgets = [50, 100, 200];
const results = {};

for (const b of budgets) {
  const f32_ms = benchPlayer(moka.WasmPuctPlayer, b, 1.5, 8);
  const i8_ms = benchPlayer(moka.WasmPuctPlayerInt8, b, 1.5, 8);
  const speedup = f32_ms / i8_ms;
  results[b] = { f32_ms, i8_ms, speedup };
  console.log(`  b${b}: f32=${f32_ms.toFixed(1)}ms  int8=${i8_ms.toFixed(1)}ms  speedup=${speedup.toFixed(2)}x`);
}

console.log('');
console.log('=== Summary ===');
const b50 = results[50];
if (b50) {
  console.log(`  b50: ${b50.f32_ms.toFixed(1)}ms → ${b50.i8_ms.toFixed(1)}ms (${b50.speedup.toFixed(2)}x)`);
  if (b50.i8_ms < 30.0) {
    console.log(`  ✓ int8 b50 = ${b50.i8_ms.toFixed(1)}ms is BELOW the 30ms floor!`);
  } else {
    console.log(`  ✗ int8 b50 = ${b50.i8_ms.toFixed(1)}ms is still above 30ms`);
  }
}
