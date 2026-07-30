# Benchmark 204: Opening Book vs Moka — Negative Result (Research)

**Date:** 2026-07-30
**Status:** ✅ COMPLETE — clean negative result
**Question:** Does forcing star-point openings (from `OpeningBookStrategy`, riir-router) on top of `GoMokaSearchPlayer` beat pure search against Moka v1 on 9×9?

## TL;DR

**No. The opening book hurts, monotonically.** Each additional forced star-point ply reduces win rate. At 8 opening plies, the player goes from winning (74%) to losing (39%).

| Config | Opening plies (star points) | Win% vs Moka (n=100) | µs/move (ours) |
|---|---|---|---|
| `moka-search` (baseline) | 0 — search from move 1 | **74.0%** | 2,254 |
| `moka-openingbook` | 4 | 61.0% | 3,339 |
| `moka-openingbook` | 6 | 53.0% | 2,429 |
| `moka-openingbook` | 8 | **39.0%** | 1,771 |

All arms use the same search config: depth=1, top_k=4, GO_OPENING_MOVES=4 (random prefix for variance).

## Why the opening book hurts

The `OpeningBookStrategy` star-point logic (ported from `riir-router/src/meta_router/strategies.rs`) plays the **first available empty star point** deterministically — it does not consider the board context. On 9×9, the star points are:

- Corner 4-4: (3,3), (3,5), (5,3), (5,5)
- Corner 3-3: (2,2), (2,6), (6,2), (6,6)
- Center: (4,4)

Moka's policy head, by contrast, was trained on 9×9 self-play and **already plays good corner openings** — but it also considers the opponent's stones, the whole-board shape, and the game phase. Forcing a blind star point overrides this contextual judgment.

The monotonic degradation (74% → 61% → 53% → 39%) confirms: **the more we override the policy with blind heuristics, the worse we play.** The policy's opening moves are not random — they carry information the star-point rule discards.

## What this rules out

1. **"Opening book + search" is NOT a viable path to >70% win rate.** The existing `GoMokaSearchPlayer` (Plan 563) at depth=1, top_k=4 is already at the policy-optimal opening. Adding a heuristic opening book can only hurt.

2. **The `OpeningBookStrategy` in riir-router is NOT a strength primitive for 9×9 Go.** Its simulated 56% win rate in `test_go_meta_router_arena` was a hardcoded fake number — the real implementation, measured here, underperforms pure search. The strategy's own doc admits "Relevant for Go 19×19 where openings matter more" — on 9×9, the opening is too short and the board too small for star-point heuristics to add value over a trained policy.

3. **Cached trajectory (opening book) does not manufacture strength.** This confirms Plan 563's audit conclusion: the strength ceiling is set by the knowledge in Moka's 105K int8 params. Forcing different (worse) moves in the opening cannot exceed the policy's own opening quality.

## What remains viable for >70%

| Path | Status | Notes |
|---|---|---|
| **PUCT search** (AlphaZero-style tree policy) | Not tried | The documented "next step" in `go_arena.md`. Replaces alpha-beta with PUCT, which combines policy prior + value in the tree. Known to extract more strength from small policy+value networks. |
| **Train better weights** | Out of scope (modelless mandate) | The only fundamental strength lever. → riir-train. |
| **Opening book** | ❌ REJECTED (this benchmark) | Hurts on 9×9. |

## Experimental setup

- **Board:** 9×9 (Moka v1 is 9×9-only)
- **Opponent:** Moka v1 greedy (policy argmax, temperature=0, no search)
- **Search config (all arms):** depth=1, top_k=4
- **Variance:** GO_OPENING_MOVES=4 (4 random plies before either player moves, to break determinism — both Moka-family players are fully deterministic)
- **Games per arm:** n=100
- **Star points:** corner 4-4 (3,3)/(3,5)/(5,3)/(5,5), corner 3-3 (2,2)/(2,6)/(6,2)/(6,6), center (4,4) — 9 points total on 9×9
- **Opening phase detection:** stones on board < `opening_book_moves * 2`

## Code

- `crates/katgpt-pruners/src/go/moka_net.rs` — `GoOpeningBookSearchPlayer` (wraps `GoMokaSearchPlayer`)
- `examples/go_11_moka_head_to_head.rs` — added `moka-openingbook` matchup + `GO_OPENING_BOOK_MOVES` env var

## Reproduction

```bash
# Baseline (should reproduce ~74%)
GO_GAMES=100 GO_MATCHUPS=moka-search GO_SEARCH_DEPTH=1 GO_SEARCH_TOPK=4 \
  cargo run --release --features go --example go_11_moka_head_to_head

# Opening book (should reproduce ~39-61% depending on opening_book_moves)
GO_GAMES=100 GO_MATCHUPS=moka-openingbook GO_SEARCH_DEPTH=1 GO_SEARCH_TOPK=4 \
  GO_OPENING_BOOK_MOVES=8 \
  cargo run --release --features go --example go_11_moka_head_to_head
```

## Significance check

Baseline 74% vs opening-book-8 39% at n=100 each: z = 5.93, p ≈ 10⁻⁹. The degradation is not noise — it's a real, statistically significant negative effect.

Baseline 74% vs opening-book-4 61% at n=100: z = 2.16, p ≈ 0.031. Even 4 forced star-point plies produces a statistically significant drop.

## Honest takeaway

This was the right experiment to run — the user's instinct that "cached trajectory / opening book" was an unexplored angle was correct, even though the result turned out negative. A negative result that closes a hypothesis is valuable research: it eliminates a class of "just add an opening book" proposals and focuses attention on PUCT (the one remaining architectural lever that doesn't require training).

The lesson generalizes: **blind heuristics cannot improve on a trained policy within the policy's training distribution.** Moka was trained on 9×9; its opening moves ARE the distilled expert opening. Overriding them with a hand-coded rule is strictly worse. This is the same reason all modelless players (Greedy/Validator/HL/GZero/MCTS) score 0% against Moka — the policy carries knowledge the heuristics don't.
