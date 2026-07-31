// bench_int8_dot_206.js — WASM int8×int8 vs f32×f32 dot product benchmark (Issue 206)
//
// Tests whether the WASM extmul-based int8 dot kernel delivers a speedup
// over the f32 dot kernel under V8 JIT. Native aarch64 SDOT showed 2.5-6.3×
// per-dot speedup (Bench 565); this verifies the WASM path.
//
// Build the wasm first:
//   ./scripts/build-moka-wasm.sh --nodejs
//   node crates/katgpt-moka-wasm/bench/bench_int8_dot_206.js

const path = require('path');
const GLUE = path.join(__dirname, '..', 'target-wasm', 'nodejs', 'katgpt_moka_wasm.js');

let moka, ex;
try {
  moka = require(GLUE);
  ex = moka.__raw_wasm;
} catch (e) {
  console.error('error: cannot load wasm glue at ' + GLUE);
  console.error('  did you run ./scripts/build-moka-wasm.sh --nodejs first?');
  console.error('  (' + e.message + ')');
  process.exit(1);
}

// Check exports exist
if (!ex.bench_dot_f32 || !ex.bench_dot_i8) {
  console.error('error: bench_dot_f32 / bench_dot_i8 exports not found.');
  console.error('  Ensure the wasm was built with simd128 enabled.');
  process.exit(1);
}

// Representative Moka conv dot sizes
const SIZES = [16, 32, 108, 144, 162, 288, 324];
const ITERS = 500000;
const WARMUP = 10000;

function allocF32(n) {
  const ptr = ex.wasmi_alloc(n);
  const view = new Float32Array(ex.memory.buffer, ptr, n);
  return { ptr, view };
}

function allocI8(n) {
  const ptr = ex.wasmi_alloc(n);
  const view = new Int8Array(ex.memory.buffer, ptr, n);
  return { ptr, view };
}

// Fill with deterministic pseudo-random data
function fillF32(view) {
  for (let i = 0; i < view.length; i++) {
    view[i] = Math.sin(i * 0.1) * 1.7 + Math.cos(i * 0.3) * 0.8;
  }
}

function fillI8(view) {
  let maxAbs = 0;
  const f32tmp = new Float32Array(view.length);
  for (let i = 0; i < view.length; i++) {
    f32tmp[i] = Math.sin(i * 0.1) * 1.7 + Math.cos(i * 0.3) * 0.8;
    maxAbs = Math.max(maxAbs, Math.abs(f32tmp[i]));
  }
  const invScale = 127.0 / Math.max(maxAbs, 1e-30);
  for (let i = 0; i < view.length; i++) {
    let q = Math.round(f32tmp[i] * invScale);
    q = Math.max(-128, Math.min(127, q));
    view[i] = q;
  }
}

console.log('╔══════════════════════════════════════════════════════════════════╗');
console.log('║  WASM int8×int8 vs f32×f32 dot (Issue 206, Node V8 JIT)        ║');
console.log('╚══════════════════════════════════════════════════════════════════╝');
console.log('');
console.log('┌──────┬──────────────┬──────────────┬──────────┐');
console.log('│ size │ f32 ns/dot   │ i8 ns/dot    │ speedup  │');
console.log('├──────┼──────────────┼──────────────┼──────────┤');

let dotPasses = 0;

for (const size of SIZES) {
  const a_f32 = allocF32(size);
  const b_f32 = allocF32(size);
  fillF32(a_f32.view);
  fillF32(b_f32.view);

  const a_i8 = allocI8(size);
  const b_i8 = allocI8(size);
  fillI8(a_i8.view);
  fillI8(b_i8.view);

  // Warm up f32
  for (let i = 0; i < WARMUP; i++) {
    ex.bench_dot_f32(a_f32.ptr, b_f32.ptr, size, 1);
  }
  // Measure f32
  const t0 = process.hrtime.bigint();
  for (let i = 0; i < ITERS; i++) {
    ex.bench_dot_f32(a_f32.ptr, b_f32.ptr, size, 1);
  }
  const f32_ns = Number(process.hrtime.bigint() - t0) / ITERS;

  // Warm up i8
  for (let i = 0; i < WARMUP; i++) {
    ex.bench_dot_i8(a_i8.ptr, b_i8.ptr, size, 1);
  }
  // Measure i8
  const t1 = process.hrtime.bigint();
  for (let i = 0; i < ITERS; i++) {
    ex.bench_dot_i8(a_i8.ptr, b_i8.ptr, size, 1);
  }
  const i8_ns = Number(process.hrtime.bigint() - t1) / ITERS;

  const speedup = f32_ns / i8_ns;
  if (speedup >= 1.5) dotPasses++;

  console.log(
    `│ ${String(size).padStart(4)} │ ${f32_ns.toFixed(1).padStart(12)} │ ${i8_ns.toFixed(1).padStart(12)} │ ${speedup.toFixed(2).padStart(7)}× │`
  );
}

console.log('└──────┴──────────────┴──────────────┴──────────┘');
console.log('');
console.log(`T4 (WASM dot ≥1.5× at any size): ${dotPasses > 0 ? '✅ PASS (' + dotPasses + '/' + SIZES.length + ')' : '❌ FAIL'}`);
console.log('');

if (dotPasses > 0) {
  console.log('VERDICT: WASM int8 extmul delivers a speedup. Proceed to conv2d_int8.');
} else {
  console.log('VERDICT: WASM int8 extmul does not clear the 1.5× gate.');
  console.log('         The extmul approach (7 instrs/16 muls) is not enough without');
  console.log('         the native i8x16.dot_s instruction (missing from stable Rust).');
  console.log('         Wait for Rust stdarch to expose i32x4_dot_i8x16_s, or use');
  console.log('         nightly + the intrinsic path.');
}
