# Issue 204: Port PUCT into `katgpt-moka-wasm` + measure the combined build

**Date:** 2026-07-31
**Status:** ✅ RESOLVED (code + measurements landed)
**Kind:** measurement gap (POC/proof task)
**Blocks:** closes the `go_arena.md` Table-A-vs-Table-B non-combination footnote

## The gap (what's wrong today)

`.docs/06_game_arenas/go_arena.md` ships two tables that are deliberately NOT
combined:

- **Table A (native strength):** PUCT search reaches 98.0% win vs Moka at
  budget=200, 81.1 ms/move (native NEON).
- **Table B (browser speed):** our WASM forward pass is 0.5 ms/move (10.7×
  faster than real Moka) — but this is **greedy forward pass only**, no
  search.

The doc admits in a footnote ("Nothing in table A has been ported to WASM")
that **no single build is both 98% strong and fast** — but never measures the
combined build that would settle it. The two-table split reads as defensive
framing: it preserves the flattering "10.7× faster" headline while sidestepping
the fact that combining PUCT + WASM inverts *both* headlines (speed advantage
evaporates; only the strength advantage remains, and at a much higher latency
than Moka greedy).

## The projection (why it was likely avoided)

Back-of-envelope from the measured pieces:

- WASM SIMD forward pass: **0.5 ms** (real Chrome, measured)
- PUCT-200 ≈ 200 forward passes + tree overhead per move
- Native PUCT-200: 81.1 ms ⇒ ~0.40 ms/pass amortized (NEON)
- WASM per-pass cost is comparable to native, **plus** the non-SIMD tree
  overhead (state clones, `Vec<PuctNode>` growth, softmax prior) which is
  worse in WASM than native

⇒ projected WASM+PUCT-200 ≈ **100–150 ms/move**, ~4–23× slower than real
Moka's 6.4 ms greedy, while winning ~94–98%. The projection could be wrong
in either direction — WASM's JIT may handle Vec-heavy tree code better or
worse than estimated — which is itself an argument for measuring rather than
projecting.

## The fix (this issue)

Port `GoPuctMokaPlayer` from `katgpt-pruners/src/go/moka_net.rs` into
`katgpt-moka-wasm`, adapted to that crate's standalone `Board` (no
`katgpt-core` / `GoState` / `GoPlayer` deps). Then measure the combined
build in both:

1. **wasmi** (interpreted, no JIT) — mirrors the existing forward-pass
   `wasmi_infer_latency` test; gives an upper bound.
2. **real Chrome via Playwright** — the headline number, apples-to-apples
   with Table B's 0.5 ms forward-pass figure.

Then replace the doc's footnote with a real row.

## Scope (concrete)

- [x] Extend `Board`: add `consecutive_passes: u8` (track in `play`/`pass`),
      `is_game_over()`, `area_score(color)` for terminal reward.
- [x] Port `PuctNode` + search loop into `katgpt-moka-wasm/src/puct.rs`
      (adapt `GoState`→`Board`, `GoAction`→`Option<usize>` action).
- [x] Add raw C-ABI `wasmi_puct_search` export (mirrors `wasmi_infer`).
- [x] Add wasm-bindgen `WasmPuctPlayer` export for the browser harness.
- [x] Add `wasmi_puct_latency` test (budget=50/100/200).
- [x] Add `bench_puct.html` browser harness; run in real Chrome via
      Playwright for the headline number.
- [x] Update `go_arena.md` Table B footnote → real Table C with measured latency.
- [x] Update `.benchmarks/205_puct_search_vs_moka_win.md` with a WASM column.

## Results (the headline)

Real Chrome via Playwright, mid-game fixture (8 stones), n=10 moves/config:

| Config | Median ms/move | Avg nodes/move | ms/node |
|---|---|---|---|
| PUCT b50  | **29.8**  | 50  | 0.594 |
| PUCT b100 | **59.4**  | 100 | 0.594 |
| PUCT b200 | **118.1** | 200 | 0.591 |

wasmi upper bound (interpreted, no JIT): b50=1,288 ms, b100=2,744 ms,
b200=5,462 ms per move.

**The projection was right.** Combined PUCT+WASM-200 is ~118 ms/move — ~18.5×
slower than real Moka greedy (6.4 ms), while winning ~94–98%. The "10.7×
faster" headline from Table B survives ONLY at greedy strength (where we
lose to Moka). Combining the two good things inverts the speed headline;
only the strength advantage remains. The honest framing replaces the prior
defensive two-table split — see `go_arena.md` Table C.

## Honest scope note

A full **win-rate** measurement (does WASM+PUCT actually win ~94-98% vs
Moka greedy in-browser?) requires a complete arena harness with two players,
terminal scoring, and win detection — substantially larger than this issue.
This issue measures **latency** (the Table-B axis) for the combined build,
which is what closes the framing gap. Win-rate parity is structurally
covered by the native PUCT result (same algorithm, same weights, same
feature encoder — only the board wrapper differs) but is not re-measured
in-browser here.
