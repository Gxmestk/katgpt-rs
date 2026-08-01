# katgpt-rs: Go Arena — AI vs AI Auto-Play Engine (Plan 065)

## Overview

A full Go (Baduk/Weiqi) game engine in pure Rust with 6 AI player strategies, Tromp-Taylor area scoring, and headless tournament infrastructure. Supports 9×9, 13×13, and 19×19 boards with komi, ko rule, and suicide prevention.

The arena serves as a fourth integration test bed for the HL thesis: **bandit-driven action selection + template-guided exploration > static heuristics or random baselines** in a deterministic perfect-information game.

Feature flag: `go = ["bandit", "dep:reqwest"]` (reqwest for AutoGo API bridge).

## Architecture

### Game Engine (`src/pruners/go/`)

| Module | Purpose |
|--------|----------|
| `types.rs` | `GoAction`, `GoCell`, `GoFrozenBandit`, `GoFrozenTemplates` |
| `state.rs` | `GoState` board state, legal moves, advance, Tromp-Taylor scoring, `GoHeuristic`, `GameState` trait impl |
| `players.rs` | `GoPlayer` trait + 6 AI player implementations (Random, Greedy, Validator, HL, GZero, MCTS) |
| `autogo_client.rs` | REST API client for external Go engines |
| `replay.rs` | Game recording and deterministic playback (`GoReplay`, `MoveRecord`) |
| `tournament.rs` | Head-to-head tournament runner against AutoGo agents |
| `g_zero_player.rs` | G-Zero self-play with HintDelta and absorb-compress |
| `autoresearch.rs` | AutoResearch loop for automated hyperparameter search |
| `analytics.rs` | Game analytics, sample computation, replay conversion |
| `event_log_player.rs` | Event-sourced game traces with fork-and-diff (feature: `event_log`) |

### Core Types

```text
GoState
  ├─ board: Vec<GoCell>        // Black, White, Empty
  ├─ size: usize               // 9, 13, or 19
  ├─ to_play: GoCell           // current player
  ├─ ko_point: Option<usize>   // simple ko rule
  ├─ consecutive_passes: u8    // game ends at ≥ 2
  ├─ move_count: u32           // total moves (including passes)
  ├─ komi: f32                 // compensation (default 7.5)
  ├─ captured_black: u32       // stones captured BY Black
  └─ captured_white: u32       // stones captured BY White

GoAction
  ├─ Place(usize, usize)       // stone placement (row, col)
  └─ Pass                      // pass turn

Scoring: score() returns f32 (black_score - white_score, komi included)
  get_winner() returns Option<GoCell> (None = draw)
```

### Scoring: Tromp-Taylor Area

Tromp-Taylor area scoring counts stones on the board plus empty points surrounded entirely by one color. This is the standard for computer Go — simple, deterministic, and no need for life/death judgment.

```text
score = own_stones + empty_points_reachable_only_by_own_color
winner = if black_score > white_score + komi then Black else White
```

## Player Strategies (6 Tiers)

### T1: Random

Randomly selects from legal moves. Baseline only.

```text
Strategy: uniform random from legal_moves()
Win rate vs self: ~30% (first-player disadvantage from komi)
```

### T2: Greedy

Scores every legal move by captures + liberties + positional heuristics.

```text
Scoring weights (Plan 073 — corner priority heuristic):
  capture_value:      10.0 per captured stone
  liberty_value:       1.0 per liberty after placement
  corner_side_bonus:   3.0 for 3rd line (territory), 2.0 for 4th line (influence)
  edge_penalty:       -2.0 for 1st line (too close to edge)
  connect_bonus:       1.0 per adjacent own stone, 0.5 per diagonal (bamboo joint)
  isolation_penalty:  -1.0 if isolated in enemy territory (≥2 adjacent opponent)
  random_noise:        0.1 for tie-breaking
Win rate vs Random: 70% (14/20 games) — more positional, less greedy on captures
```

### T3: Validator

Greedy + deterministic safety rules to avoid obviously bad moves.

```text
Safety rules:
  - No self-atari (placing into own group with 1 liberty)
  - No filling own eyes (surrounded by 4+ own stones)
  - Avoid 1-point jumps into enemy territory
Win rate vs Random: 100% (10/10 games)
```

### T4: HL (Heuristic Learning)

Bandit Q-learning over 8 move categories with AbsorbCompress.

```text
Categories: CornerStar, SideApproach, CenterControl, Capture, Defend, Extend, Influence, Pass
Strategy: UCB1 over category Q-values → pick best legal move in category
Adaptation: Q-values update per game outcome, cross-game learning
Win rate vs Random: 100% (10/10 games)
```

### T5: GZero (Template UCB1)

Template-based action proposal + bandit selection + delta-gating absorb-compress.

```text
Templates: CornerStar, Capture, Defend, Tenuki
Selection: UCB1 over template bandit arms
Delta-gating: only promote templates with positive δ (outcome improvement)
Self-play: 500 episodes, learns template ranking over time
```

### T6: MCTS (Monte Carlo Tree Search)

Generic MCTS with `GoState::advance()` as forward model.

```text
Configurable budget: 50–1000 simulations per move
Rollout: random playout to terminal, then Tromp-Taylor scoring
Selection: UCB1 tree policy
Weakness: budget=200 insufficient for Go's ~80 branching factor
Win rate vs Random: 65% (13/20 games, improved from 55% via territorial heuristic)
```

## Examples & Results

### go_00_api_bridge

REST API client for playing against an external AutoGo server.

**Status:** Requires running AutoGo server (`scripts/autogo_server.sh`). Plays random games against server agents via HTTP.

```bash
# Start AutoGo server first
./scripts/autogo_server.sh

# Run bridge
cargo run --features go --example go_00_api_bridge
```

---

### go_01_mcts — MCTS vs Random Benchmark

MCTS (budget=200) vs Random, 20 games on 9×9, komi=7.5.

**Results:**

| Metric | Value |
|--------|-------|
| MCTS Win Rate | 65% (13W / 7L) |
| As Black | 60% |
| As White | 70% |
| Avg Moves/Game | ~185 |
| Avg Time/Game | ~0.18s |
| Moves/sec | ~1018 |

**Verdict:** MCTS beats Random at budget=200 after Plan 073 territorial heuristic fix (65%, up from 55%). Previous 55% was caused by backwards center-preference weight — Go rewards corner/edge territory, not center control. Budget=200 still insufficient for Go's ~80 branching factor; higher budgets needed for stronger play.

```bash
cargo run --features go --example go_01_mcts
```

---

### go_02_tournament — All Players vs Random

Round-robin tournament: each player vs Random, 10 games, 9×9.

**Results:**

| Player | vs Random Win% | Avg Moves | Avg Time |
|--------|---------------|-----------|----------|
| Random | 35% | ~291 | <0.1s |
| Greedy | 70% | ~302 | 0.2s |
| Validator | **100%** | ~302 | 0.2s |
| HL | **100%** | ~302 | 0.5s |
| MCTS | 80% | ~196 | 0.2s |

**Verdict:**
- Validator/HL dominate Random with 100% win rate
- Greedy dropped to 70% after Plan 073 — more positional play, sometimes misses tactical captures
- MCTS improved from 55% → 80% via territorial heuristic in tournament format
- Random baseline wins ~35% due to first-player advantage from komi (Black)

```bash
cargo run --features go --example go_02_tournament
```

---

### go_03_head_to_head — AutoGo Server Tournament

Head-to-head matchups against external Go engines (e.g., GNU Go) via AutoGo REST API.

**Status:** Requires running AutoGo server. Tests Random, Greedy, HL, GZero, MCTS against server agents.

```bash
# Start AutoGo server
cd autogo && python play.py

# Run with minimal games for quick test
GO_GAMES=2 cargo run --features go --example go_03_head_to_head
```

---

### go_04_gzero — G-Zero Self-Play

GZero template-based self-play with delta-gating absorb-compress + adaptive komi + swap-colors (Plan 091). 500 episodes on 9×9.

**Results (adaptive komi, initial=7.5):**

| Metric | Value |
|--------|-------|
| Total Episodes | 500 |
| Duration | 3.1 min |
| Episodes/sec | 2.7 |
| Black Wins | 493 (98.6%) |
| White Wins | 7 (1.4%) |
| Draws | 0 |
| Final Komi | 41.9 (converged from 7.5) |
| Avg Moves/Game | 243 |

**Komi Convergence Curve (score-margin-guided with damping):**

| Episode | Komi | Avg Score Margin | B Win Rate (window) |
|---------|------|-----------------|---------------------|
| 50 | 17.5 | +30.2 | 92% |
| 100 | 27.5 | +22.8 | 96% |
| 150 | 34.3 | +13.6 | 98% |
| 200 | 38.2 | +7.7 | 100% |
| 250 | 40.1 | +3.8 | 100% |
| 300 | 41.0 | +1.9 | 100% |
| 350 | 41.5 | +1.0 | 100% |
| 400 | 41.8 | +0.5 | 100% |
| 450 | 41.9 | +0.2 | 100% |
| 500 | 41.9 | ~0 | 100% |

**Results at pre-converged komi=42 (150 episodes, separate run):**

| Metric | Value |
|--------|-------|
| Black Wins | 121 (80.7%) |
| White Wins | 7 (4.7%) |
| Draws | 22 (14.7%) |
| Final Komi | 41.0 (stable ±1) |

**Template δ Ranking (fixed komi=7.5, no adaptive):**

| Rank | Template | δ |
|------|----------|---|
| 🥇 | Capture | +0.0000 |
| 🥈 | CornerStar | -7.25 |
| 🥉 | Tenuki | -9.50 |
| 4 | Defend | -50.22 |

**Verdict:**
- Adaptive komi converges from 7.5 → ~42 in 300 episodes (score-margin-guided, damped)
- Template-based bots on 9×9 need ~42 komi (vs 7.5 for pros) — weak play amplifies first-move advantage by ~5.6×
- At komi=42: draws appear (14.7%), confirming equilibrium; win rate narrows to ~81/5/14 B/W/D
- Cumulative stats still show 98.6% Black due to low-komi convergence phase (episodes 1-250)
- For production: pre-converged komi=42 recommended as `initial_komi` to skip convergence phase
- Absorb-compress: no templates promoted (δ below threshold for all templates)
- Capture remains the only neutral-δ template (safe to play)
- **Swap-colors (Option B)** enabled by default — each agent plays both sides equally (even eps: A=Black, odd eps: A=White), giving per-agent win rate balance toward ~50%

```bash
# Full self-play with adaptive komi
cargo run --features go --example go_04_gzero

# Quick demo (10 episodes)
GO_SET=quick cargo run --features go --example go_04_gzero
```

---

### go_05_autoresearch — AutoResearch Hyperparameter Scan

Bandit-driven hyperparameter search. 10 arms (configs), 50 evaluations, 10 games/eval, Greedy player vs Random baseline.

**Config Space:**

| Param | Range |
|-------|-------|
| MCTS Budget | 0 (Greedy only) |
| Rollout Depth | 10–50 |
| Exploration C | 0.9–1.7 |
| Bandit ε | 0.11–0.47 |
| Templates | 2–4 |

**Results:**

| Metric | Value |
|--------|-------|
| Best Config | M0:D30:C1.7:E0.11:T4 |
| Best Win Rate | 100% |
| Total Arms | 10 (all active) |
| Total Games | 500 |
| Duration | 62.9s (8 games/s) |
| Convergence | STABLE (-1.4pp Q1→Q4) |

**Top 5 Arms:**

| Rank | Config | Win Rate |
|------|--------|----------|
| 1 | M0:D20:C1.7:E0.32:T3 | 100% |
| 2 | M0:D20:C1.0:E0.14:T4 | 100% |
| 3 | M0:D50:C0.9:E0.21:T4 | 100% |
| 4 | M0:D50:C0.9:E0.15:T3 | 100% |
| 5 | M0:D50:C1.4:E0.41:T2 | 100% |

**Verdict:** All configs beat Random since they use Greedy baseline. AutoResearch shows the bandit correctly identifies that Random is an easy opponent — no meaningful config differentiation yet. Needs harder opponents (Validator, HL) to find meaningful hyperparameter differences.

```bash
cargo run --features go --example go_05_autoresearch
```

---

### go_06_bench — Go Benchmark Suite

Comprehensive benchmark: advance performance, MCTS throughput, player scaling laws.

#### T43: GoState::advance() Performance

| Config | Legal Moves | ops/sec | µs/advance | µs/clone |
|--------|-------------|---------|------------|----------|
| 9×9 opening | 82 | 486,773 | 2.05 | 1.87 |
| 9×9 midgame | 53 | 432,855 | 2.31 | 1.66 |
| 9×9 endgame | 11 | 390,964 | 2.56 | 1.81 |
| 19×19 opening | 362 | 129,442 | 7.73 | 8.18 |
| 19×19 midgame | 312 | 132,774 | 7.53 | 11.57 |
| 19×19 endgame | 169 | 122,326 | 8.17 | 7.42 |

#### T44: MCTS Search Throughput (9×9, ~10 moves played)

| Budget | µs/search | actions/sec | nodes/sec |
|--------|-----------|-------------|-----------|
| 50 | 316 | 3,168 | 158,408 |
| 200 | 1,439 | 695 | 138,982 |
| 500 | 3,291 | 304 | 151,915 |
| 1000 | 6,920 | 145 | 144,504 |

#### T46: Player Scaling Laws (9×9, 20 games each, Plan 073)

| Player | Wins | Losses | Win Rate |
|--------|------|--------|----------|
| Random | 7 | 13 | 35.0% |
| Greedy | 14 | 6 | 70.0% |
| Validator | 20 | 0 | **100.0%** |
| HL | 20 | 0 | **100.0%** |
| MCTS(200) | 17 | 3 | **85.0%** |

**Verdict:**
- `advance()` is fast: 2–3µs on 9×9, 7–8µs on 19×19 (2.5× faster than Plan 075 baseline)
- Clone cost ≈ advance cost (both copy the board vector)
- MCTS throughput scales linearly with budget (~140K nodes/sec, 5.6× faster than baseline)
- MCTS win rate improved from 60% → 85% via territorial heuristic
- Greedy more positional at 70% — trades some tactical wins for better shape

```bash
cargo run --features go --example go_06_bench
```

---

### go_07_tui — AI vs AI Auto-Play TUI

Animated ratatui TUI replay with unicode stone rendering. Two-panel layout: board grid + scoreboard.

```bash
# Default: Greedy (Black) vs Validator (White) on 9×9
cargo run --features go --example go_07_tui

# Custom players and board
cargo run --features go --example go_07_tui -- --black hl --white gzero --size 9

# Custom seed
cargo run --features go --example go_07_tui -- --seed 99
```

**Controls:** ←/→ step, Space auto-play (300ms), R new game, Home/End jump, Q quit.

**Rendering:** 1-char-wide symbols: `●` (black), `○` (white), `·` (empty), `+` (star/hoshi), `x` (ko). Last move highlighted green+bold.

## Cross-Domain Comparison

| Domain | Engine | ECS | Best AI | vs Random | Key Metric |
|--------|--------|-----|---------|-----------|------------|
| Bomberman | Tick-based | bevy_ecs | HL (bandit) | ~4:1 score | Survival 4% |
| Monopoly | FSM events | bevy_ecs | HL (bandit) | 56.5% win | Survival 93.7% |
| FFT Tactics | ATB queue | — | HL (bandit) | Enemy 93% | Unit MVP: Knight-HL 176 kills |
| **Go** | **Turn-based** | — | **Greedy/HL** | **100%** | **MCTS needs 1000+ budget** |

## Key Findings

1. **Territorial heuristic fixed (Plan 073)** — Go rewards corner/edge territory, not center control. Flipping `center_preference()` to `territorial_preference()` with phase-aware evaluation (Early: corners, Late: influence) improved MCTS from 55% → 65% vs Random.

2. **MCTS modest at budget=200** — Go's branching factor (~80 on 9×9, ~250 on 19×19) exhausts budget. The corrected heuristic helps (55% → 65%), but budget=200 remains insufficient for reliable dominance.

3. **Greedy trades tactics for position** — After Plan 073, Greedy dropped from 100% → 70% vs Random. The corner/side bonus makes it play more positionally, sometimes missing tactical captures. Validator and HL (which layer on top of Greedy) remain at 100%.

4. **Self-play komi imbalance fixed (Plan 091)** — GZero self-play originally showed 98.6% Black wins at komi=7.5. Adaptive score-margin-guided komi converges to ~42 (5.6× the pro komi), narrowing B/W to ~81/5/14 with 14.7% draws. **Swap-colors (Option B)** ensures each agent plays both sides equally, achieving per-agent ~50% win rate balance. Production runs should use `initial_komi=42` on 9×9 with `swap_colors=true`.

5. **HL adapts but doesn't surpass Greedy** — HL's bandit learning reaches 100% vs Random (matching Validator), but hasn't been tested head-to-head against Greedy yet. The TUI makes this easy to observe.

6. **advance() is production-ready** — 2µs on 9×9, 8µs on 19×19 (2.5× faster than initial). MCTS at budget=1000 processes ~144K nodes/sec, enabling real-time play on 9×9.

## Bug Fixes Discovered

### bomber_05_replay_gen.rs & bomber_06_replay_gen_v2.rs — Index Out of Bounds

`BomberAction` enum has 7 variants (Up, Down, Left, Right, Bomb, Wait, Detonate) but `ACTION_NAMES` and `action_counts` arrays had only 6 elements. When `Detonate` action (index 6) was selected, `action_counts[6]` panicked with "index out of bounds: the len is 6 but the index is 6".

**Fix:** Changed arrays from `[T; 6]` to `[T; 7]` and added `"Detonate"` to `ACTION_NAMES`.

## go_11_moka_head_to_head — Moka v1 (real weights) Head-to-Head (Plan 563)

**Question:** can our modelless Go players (heuristic/bandit, zero learned parameters) beat [Moka v1](https://million.dev/moka) — a real, MIT-licensed, 105,353-parameter 9×9 Go policy/value network (`github.com/millionco/moka`, distilled from KataGo, self-reported ~2 kyu, 30% win rate vs KataGo b6c96) — on strength, size, or speed?

**Method:** Moka's real weights (`go-model.bin`, 113,648 bytes, sha256-verified against the upstream manifest) were vendored and its exact architecture (`MokaGlobalResidualNetwork` — 3×3 stem, 12 nested-bottleneck residual blocks with 3 global-pooling branches, policy+value heads) reimplemented natively in Rust — `crates/katgpt-pruners/src/go/moka_net.rs` — with no Python, Node, WASM, or browser involved. Correctness was validated by an independent hand-derived closed-form check of the stem layer on an empty board (re-parsing the manifest via generic JSON, not the loader code under test) plus a finiteness/shape smoke test, both passing (`cargo test -p katgpt-pruners --features go --lib go::moka_net`). Moka plays greedy policy argmax (`temperature=0.0`, no search), matching its own arena convention.

**Result (N=20 games/matchup, 9×9, komi=7.0 per Moka's own training convention, alternating colors):**

| Player | W/L vs Moka | Win% | Avg moves | µs/move (ours) | µs/move (Moka) |
|---|---|---|---|---|---|
| Greedy | 0W/20L | 0.0% | 121.5 | 207.8 | 3378.0 |
| Validator | 0W/20L | 0.0% | 126.0 | 104.7 | 3388.2 |
| HL | 0W/20L | 0.0% | 111.3 | 240.5 | 3379.7 |
| GZero | 0W/20L | 0.0% | 115.5 | 132.4 | 3385.0 |
| MCTS (budget=200) | 0W/20L | 0.0% | 109.2 | 2436.3 | 3402.6 |

Margins are large — typically 35–80 points on an 81-point board (e.g. `W+55.5`, `B+68.5`), not close games. Reference only, not run by this harness: Gemma 2 2B as a Go player (`riir-ai` Plan 393/408/410) scored 0% gain over the random baseline at ~50 s/move CPU — exhaustively benchmarked separately, cited here for scale, not re-run.

**Verdict:**

- **Strength: no.** Every modelless player loses every game to Moka's real weights, decisively. A 105K-parameter distilled CNN with real conv/policy/value structure beats hand-crafted heuristics and a budget-200 MCTS outright — the STRATEGA finding ("domain heuristics > generic search" from Plan 056) doesn't extend to "domain heuristics > a trained-however-tiny network." This directly answers the original question: no, our modelless Go players do not currently beat Moka.
- **Size: yes, trivially, but it no longer matters.** Our heuristic players carry ~0 learned parameters (just a handful of Q-values/template deltas) vs Moka's 105,353 int8 params (113,648 B) — but a 100% loss rate makes the size "win" moot as a competitive claim.
- **Speed: mixed, and much closer than assumed.** Once both sides run natively (no browser/WASM/JS overhead on Moka's side), our heuristics are still faster per move (105–241 µs for Greedy/Validator/HL/GZero) but MCTS at budget=200 (2.4 ms) is within 1.4× of Moka's own 3.3–3.4 ms/move — a real CNN forward pass, not the µs-vs-browser-ms framing assumed before vendoring the actual weights.
- **Take-away:** the "load the real weights like we did for Gemma2" approach (this plan) produced a decisive, honest, locally-reproducible answer in under a day of work — in sharp contrast to Gemma2, where the LLM path was exhaustively benchmarked and found to have *zero* Go-playing signal at any size. Moka proves a small *trained* network beats hand-crafted heuristics on Go; Gemma2 proves a *huge* general-purpose LLM without Go-specific training does not.

```bash
cargo run --features go --example go_11_moka_head_to_head
GO_GAMES=20 GO_MCTS_BUDGET=500 cargo run --release --features go --example go_11_moka_head_to_head
```

## GoMctsMokaPlayer — MCTS + Moka's Value Head as Leaf Evaluator (negative result)

**Idea:** rather than out-heuristic a trained network from scratch, compose it with search — plug Moka's real value head into `mcts_search`'s existing pluggable `Fn(&GoState, u8) -> f32` evaluator slot (the same slot `GoMctsPlayer` uses for `GoHeuristic`), giving MCTS a genuine neural evaluator instead of random-playout-to-terminal or a hand-tuned linear formula. `MokaHeuristic` + `GoMctsMokaPlayer` (`crates/katgpt-pruners/src/go/moka_net.rs`) implement exactly this — zero additional training, pure composition of an existing generic search primitive with a frozen neural evaluator.

**Result (10 games, budget=30, rollout_depth=20; also checked at budget=150 for 4 games):**

| Config | W/L vs Moka | Win% | µs/move |
|---|---|---|---|
| MCTS-Moka, budget=30 | 0W/10L | 0.0% | 6,773 |
| MCTS-Moka, budget=150 (5×) | 0W/4L | 0.0% | 24,101 (~5× cost, as expected) |

**Still 0% — scaling the search budget 5× didn't move the needle.** Margins (52–60+ points) were statistically indistinguishable from the pure heuristics' losses.

**Why, honestly:** two things this first attempt didn't fix, both real gaps rather than a training-vs-not distinction:

1. **Rollout policy is still `RandomRolloutPolicy`** — the search selects moves uniformly at random for the ~20-ply rollout to the depth cutoff, and only THEN asks Moka's value head to judge the resulting position. Moka was trained on realistic self-play move distributions; a position reached by 20 random plies is likely out-of-distribution for its value head, so the "expert" evaluation may itself be unreliable at the leaf it's actually asked to judge.
2. **Moka's policy head isn't used at all in this version** — root-level action selection is plain UCB1 with no informed prior, so the search doesn't benefit from Moka's own move preferences, only from its position judgment after already-random play.

**Follow-up test (rules out the OOD hypothesis):** re-ran with `rollout_depth=0` — Moka's value head judges the position *immediately* after each candidate move, no random rollout at all (fully in-distribution for a value net trained on real self-play). **Still 0/10, same ~45–70 point margins, and ~40× slower** (133 ms/move at budget=80 vs 3.3 ms/move) for no gain. So the out-of-distribution rollout wasn't the real blocker.

**Actual diagnosis:** with budget ≈ branching factor (~50–80 legal moves on 9×9), each candidate move gets visited roughly *once* on average under plain UCB1 — nowhere near enough visits per arm to differentiate them, regardless of leaf-evaluation quality. This is the textbook reason AlphaZero-style search uses **PUCT** (UCB1 biased by the policy net's move probabilities) instead of plain UCB1: the policy prior is what makes search converge at practical budgets. Neither attempt above used Moka's policy head at all — only its value head.

**Next step taken:** rather than converting the generic UCB1 `mcts_search` into PUCT, the same "policy prunes / value judges" structure was implemented directly as a small alpha-beta negamax — see `GoMokaSearchPlayer` below, which is the first configuration to stop losing.

## GoMokaSearchPlayer — policy-pruned negamax (first non-losing result)

`GoMokaSearchPlayer` (`crates/katgpt-pruners/src/go/moka_net.rs`) uses **both** of Moka's heads: the policy head orders and prunes the move list (top-K beam, `Pass` included as a real candidate), the value head evaluates leaves, and alpha-beta negamax does the lookahead. This is the structure PUCT provides, reached without rewriting the shared UCB1 search. It also threads real move history through every search node (`SearchHistory`), fixing the empty-history approximation that `MokaHeuristic` above is stuck with, so feature planes 7–10 are correct at every node.

Rationale (policy improvement): raw `MokaPlayer` is policy-argmax with *zero* lookahead. Searching D plies and backing up value estimates is a strict refinement of that same policy — the standard reason search beats its own raw policy net.

**Two methodology bugs found and fixed while measuring this — both inflated/garbled earlier numbers:**

1. **Move history leaked across games.** `reset()` was only called at end-of-matchup, so each game after the first began with the previous game's trailing moves still in history, corrupting the opening's last-move feature planes. Now reset per game (safe for `GoHLPlayer`/`GoGZeroPlayer`, whose `reset()` clears only per-episode trace state — bandit Q-values still persist across games as intended).
2. **"N games" was not N samples.** Both Moka-family players are fully deterministic, so every game with the same color assignment replayed *byte-identically* — a "12 game" run was really 2 distinct games repeated 6×. Fixed with randomized opening plies (`GO_OPENING_MOVES`, default 4) — the same thing Moka's own arena does, for the same reason. Any earlier win-rate over deterministic repeats should be disregarded.

**Result (60 independent games, depth=2, top_k=8, 4 random opening plies):**

| Player | W/L vs Moka | Win% | 95% CI | exact 1-sided p | µs/move (ours) | µs/move (Moka) |
|---|---|---|---|---|---|---|
| Moka-Search (depth 2, top-K 8) | **37W/23L** | **61.7%** | [49.4%, 74.0%] | 0.046 | 119,047 | 3,379 |

**Interpretation — suggestive of a real edge, but right at the threshold, not conclusive.** Exact `P(X≥37 | n=60, p=0.5) = 0.046` — marginally significant one-sided (two-sided ≈ 0.09), and the 95% CI's lower bound (49.4%) still grazes 50%. The defensible claim: **policy+value search took us from 0% (losing every game, decisively) to a likely-but-unproven edge around 60%.** Asserting "we beat Moka" outright would overclaim on n=60; a few hundred games would settle it.

*Sampling note:* an earlier 16-game run (9W/7L, 56.2%) is a strict **prefix** of this 60-game run — same seed (42), same deterministic code path, so its first 16 games are identical. The two must NOT be pooled; the 60-game figure supersedes it.

**Cost caveat, and it matters for the original question:** ~35× Moka's per-move latency (119 ms vs 3.4 ms at the time), because every visited search node is a full forward pass. This configuration plausibly wins on *strength* while losing on *perf* — and it isn't a "modelless" win at all: it requires Moka's own weights to function. The honest summary is "search on top of their model beats their model," not "our architecture beats theirs." The latency half of that was then attacked directly — see below.

## Forward-pass kernel optimization (6.4× faster)

The 119 ms/move figure above was unacceptable, and the root cause was **our own kernel, not the search**: a single forward pass cost 3.4 ms for a 105K-param net (~5.8M MACs ⇒ only **~1.7 GMAC/s**). Three defects in `moka_net.rs`, all fixed:

1. **`for oc` was the outermost loop**, so the input was re-read `out_ch` times (32× for the stem). Now position is outermost: each position's `k*k*in_ch` neighbourhood is gathered **once** into a contiguous patch and reused across every output channel. The weight layout `[out,kh,kw,in]` means each output channel is then one contiguous dot product over that patch.
2. **~50 `vec![]` allocations per forward.** Now every layer writes into a caller-held `MokaScratch`.
3. **Single sequential f32 accumulator** — which LLVM *cannot* auto-vectorize, because FP addition isn't associative. Now `ACC_LANES = 8` independent accumulator chains, which vectorize on both AVX2 and NEON.

A 1×1-conv fast path skips the gather entirely (over half the convs in this net are 1×1).

**Measured (`GO_BENCH_FORWARD=1`, 300 iters, mid-game position):**

| Kernel | ms/pass | GMAC/s | vs baseline |
|---|---|---|---|
| Original (`for oc` outer, 1 accumulator, per-layer `vec![]`) | 3.4 | ~1.7 | 1.0× |
| Restructured + hand-rolled 8-lane accumulators | 0.535 | 10.84 | 6.4× |
| Restructured + **`katgpt_types::simd::simd_dot_f32`** | **0.392** | **14.78** | **8.7×** |

**8.7× faster overall.** Three honest notes:

- **Reusing the workspace SIMD kernel beat hand-rolling it, by 1.36×.** The first pass hand-wrote 8 independent accumulator chains and hoped LLVM would widen them. `katgpt_types::simd::simd_dot_f32` — already a dependency, already used by `symbolic_expression.rs` and `step_attribution_qualifier.rs` in this same crate — is explicit NEON / AVX2-FMA intrinsics and is simply better. This was a DRY miss worth correcting on both counts. (Length discipline matters: that kernel indexes unchecked up to `len`, so call it as `a.len().min(b.len())`, the convention the other call sites use.)
- **The allocation fix (2) was irrelevant.** Scratch reuse now measures **−1.7%** (i.e. the allocating wrapper is marginally faster, within noise). The prediction that ~50 allocs/forward mattered was simply wrong — the allocator handles that pattern for free. All of the gain came from (1) cache locality and (3) vectorization. The scratch plumbing is retained only because it removes allocator variance from search timing, not because it is a speedup.
- **Correctness is pinned, not assumed.** (3) changes FP summation order *and* introduces fused multiply-add (single rounding instead of double), so the original naive `conv2d`/`linear` are kept verbatim in the test module as an equivalence oracle (`optimized_conv_matches_naive_reference`, `optimized_linear_matches_naive_reference`, at every layer shape the net actually uses), plus `scratch_reuse_matches_fresh_forward` asserting bit-identical output across reused buffers. Without those, a numerical regression would have surfaced only as unexplained strength drift.

This speeds up **both** sides of the comparison — the baseline `MokaPlayer` port drops from 3.4 ms to ~0.54 ms/move, and the search players scale down proportionally.

### Search depth: depth 2 buys nothing over depth 1

With the kernel fixed, the remaining latency lever is search width/depth. 30 games at each setting, same seed:

| Config | W/L vs Moka | Win% | Avg moves | µs/move (ours) | µs/move (Moka) |
|---|---|---|---|---|---|
| depth=1, top_k=8 | 19W/11L | 63.3% | 79.1 | **5,211** | 567 |
| depth=2, top_k=8 | 19W/11L | 63.3% | 94.8 | 20,965 | 613 |

**Identical strength, 4× the cost.** The differing average game length (79.1 vs 94.8) confirms these are genuinely different games, not an accidentally duplicated run — the two configs play differently but end up equally strong. So the extra ply is pure waste here: at depth 1 the value head is already judging the position immediately after each candidate move, and a second ply of the *same* evaluator adds no new information it can act on.

**Combined latency result: 119 ms → 5.2 ms per move (~23×)** — 6.4× from the kernel, ~4× from dropping to depth 1. Against Moka's own 0.57 ms/move that is ~9× rather than the original ~35×, and 5.2 ms fits comfortably inside a real-time budget (e.g. a 20 Hz tick is 50 ms).

### Beam width: narrow beats wide, on BOTH axes

`top_k` turned out to be the dominant knob — and in the opposite direction to the one assumed. Depth=1, 100 games per setting, same seed:

| top_k | W/L vs Moka | Win% | Avg moves |
|---|---|---|---|
| 2 | 61W/39L | 61.0% | 76.5 |
| 3 | 69W/31L | 69.0% | 80.2 |
| **4** | **74W/26L** | **74.0%** | 80.7 |
| 6 | 68W/32L | 68.0% | 79.6 |
| 8 | 58W/42L | 58.0% | 84.8 |
| 16 | 60W/40L | 60.0% | 83.0 |

Significance (n=100 each): top_k=4 vs the 50% baseline is **z = 4.80** (p ≈ 10⁻⁶); top_k=4 vs top_k=8 is **z = 2.42** (p ≈ 0.016), so the narrow-beam advantage is real. But top_k=4 vs 3 (z = 0.78) and vs 6 (z = 0.94) are **statistically tied** — so the honest claim is *"narrow beam (3–6) beats wide beam (8–16)"*, **not** "4 is the optimum." The point estimate peaks at 4; the plateau is 3–6.

**Why narrower is stronger, not merely cheaper:** Moka's policy head is more reliable than its value head. A wide beam hands the value head authority over more candidates — including moves the policy rates poorly — and every extra candidate is another chance for value-head noise to promote a blunder above a genuinely good move. Narrowing to ~4 keeps the value head arbitrating only among moves the policy already endorses. This also retro-explains the depth-2 null result: piling on more of the *same* noisy evaluator doesn't help; **constraining its scope does.**

⚠️ **Latency numbers in the sweep above are unreliable.** Moka's own per-move latency is the control and should be constant, but it varied 404→942 µs across those runs — the machine was CPU-contended, so fine per-config latency deltas from that sweep must not be quoted. Only the coarse trend (top_k=16 clearly slowest) survives. Clean isolated figures are in the summary below.

### Strength history — every earlier figure was a mistuned or undersampled config

| n | Config | W/L | Win% | 95% CI | exact 1-sided p | status |
|---|---|---|---|---|---|---|
| 30 | depth=1, k=8 | 19W/11L | 63.3% | — | ≈0.10 | n.s., undersampled |
| 60 | depth=2, k=8 | 37W/23L | 61.7% | [49.4%, 74.0%] | 0.046 | superseded |
| 200 | depth=1, k=8 | 114W/86L | 57.0% | [50.1%, 63.9%] | 0.028 | **mistuned** (k=8) |
| 100 | depth=1, k=4 | 74W/26L | 74.0% | — | ~10⁻⁶ | small-sample high |
| **300** | **depth=1, k=4** | **210W/90L** | **70.0%** | **[64.8%, 75.2%]** | **1.7×10⁻¹²** | ✅ **quote this** |

**70.0% (CI 64.8–75.2%) is the number to quote.** Two lessons in that table:

1. **The 57% "final answer" was a tuning artifact, not the truth.** It was measured honestly at n=200 but at `top_k=8` — a config since shown to be significantly worse than `top_k=4` (z = 2.42). Sample size was never the problem there; the *configuration* was. Widening the sample on a badly-tuned config buys precision about the wrong thing.
2. **Small samples still ran high, repeatedly.** 74% (n=100) → 70% (n=300), just as 63.3% (n=30) → 57% (n=200) earlier. The direction was consistently optimistic. At n=300 with p ≈ 10⁻¹² the result is no longer in doubt.

### Bottom line on the whole exercise (superseded — see "PUCT search" below)

⚠️ **This table is the honest bottom line as of the alpha-beta result only. It was superseded same-day by PUCT (Bench 205, further down this document), which reaches 98.0% — not 70.0%. Kept here unedited for the historical record of how the investigation actually progressed; do not quote 70% as "best of ours" going forward.**

| | Moka v1 (greedy) | Best of ours *at the time* (`GoMokaSearchPlayer`, alpha-beta, depth 1, k=4) |
|---|---|---|
| Strength (9×9, n=300) | baseline | 70.0% win rate (CI 64.8–75.2%) — **since superseded by PUCT's 98.0%** |
| Params | 105,353 | same 105,353 — *it is Moka's network* |
| Weights payload | 113,648 B | same |
| Latency/move | ~0.4–0.54 ms | ~2.8 ms (~5–7×) |
| Forward pass | ~0.45 ms | ~0.45 ms (shared kernel) |

⚠️ **Timing caveat.** This machine's measurements are noisy: Moka's per-move latency — a control that should be constant — was observed anywhere from 404 to 942 µs across runs, and repeat runs of the same forward-pass bench gave 0.392 and 0.450 ms. Quote latency as approximate. In particular the `MokaScratch` reuse-vs-allocate comparison came out **−1.7% in one run and +22.7% in another**, i.e. the direction is inside the noise floor — no speedup should be claimed for the scratch plumbing (see the kernel section above; it is retained for timing stability, not for throughput).

The summary *at this point in the investigation* was: policy-pruned search on top of Moka's own weights beats greedy Moka decisively — 70% over 300 games — at roughly 5–7× the per-move cost (2.8 ms vs ~0.5 ms), ~42× better than the 119 ms/move this line of work started at. **That conclusion no longer holds as the final word** — PUCT (below) pushes strength to 98.0% at a correspondingly higher latency (~80 ms/move at budget=200), a different point on the strength/latency curve entirely, not a small refinement of this one. What still stands unchanged from this phase: our *modelless* players remain 0% against Moka (original question's answer: **no**); this configuration is not architecture-independent (needs Moka's policy and value heads); and the underlying infrastructure — parity-checked native port, 8.7× kernel speedup with equivalence guards — is what both this result and PUCT are built on.

### Investigated and rejected: Apple Neural Engine via CoreML (Issue 564)

Asked to grep harder for unused repo primitives, a deeper search surfaced `katgpt-backend::ane` — a real CoreML/ANE execution backend (not a cost model), unused for Moka. Built a scoped probe (stem + one residual block, 9 layers) rather than the full ~60-layer graph, specifically to answer the residency question before committing to the larger build.

**Result: ANE is 4.66× slower than CPU for the identical 9-layer slice** (261 µs vs 56 µs, matched workload, not a proportional estimate). Correctness was proven bit-perfect (`max_abs_diff = 5.96×10⁻⁸`, f32 epsilon) — the layout transpose work is real and correct — but the performance verdict is decisively negative. At 105K params, CoreML's fixed per-call dispatch/marshalling overhead dominates completely; the whole network only costs ~450 µs on CPU, leaving no room for a few-hundred-µs fixed overhead to amortize away even at the full graph size. Per the stop-rule set before writing any code, the remaining ~50 layers were not built. Full record: `docs/09_feature_catalog/negative_results.md` §21 (Issue 564, removed per noise-reduction rule).

**Notable asymmetry:** wins are short games with modest margins (48–82 moves, +2.5 to +18.5); losses are long games with blowout margins (116–140 moves, +41.5 to +75.5).

The tempting read is "something degrades in long games," but the causality is probably the reverse, and it's rational: at depth 2 the search *can* see `pass → opponent passes → terminal → exact score`, so when it is ahead, passing is correctly valued as a locked win and it ends the game early (short game, modest winning margin). When it is behind, passing is correctly valued as a loss, so it plays on (long game). Game length is therefore mostly an *effect* of who is ahead, not a cause of losing. The real open question is narrower and harder: why it falls behind in ~44% of games in the first place — for which the natural probes are deeper search (`GO_SEARCH_DEPTH=3`) and a wider beam (`GO_SEARCH_TOPK`), since both directly buy more of the policy-improvement effect that got us from 0% to parity.

```bash
cargo run --release --features go --example go_11_moka_head_to_head  # includes mcts-moka matchup
GO_MCTS_MOKA_BUDGET=150 GO_GAMES=4 cargo run --release --features go --example go_11_moka_head_to_head
```

### Investigated and rejected: Opening Book star-point heuristics (Bench 204)

The user correctly identified that `OpeningBookStrategy` (riir-router `meta_router/strategies.rs`) — a star-point opening book with corner 4-4/3-3, side stars, and center — was never measured against Moka. Its simulated 56% win rate in `test_go_meta_router_arena` was a **hardcoded fake number** (the test returns `0.56` by name lookup, not from real games). The real question: does forcing star-point openings on top of `GoMokaSearchPlayer` beat pure search?

**Result: the opening book hurts, monotonically.** A `GoOpeningBookSearchPlayer` wrapper (Bench 204) was built to force star points for the first N plies, then delegate to the same depth=1 top_k=4 search. n=100 per arm:

| Opening plies | Win% vs Moka |
|---|---|
| 0 (pure search, baseline) | **74.0%** |
| 4 | 61.0% |
| 6 | 53.0% |
| 8 | **39.0%** |

The degradation is monotonic and statistically significant (baseline 74% vs 8-ply 39%, z=5.93, p≈10⁻⁹). **Moka's policy already plays better openings than blind star-point heuristics** — it was trained on 9×9 and considers board context. Forcing star points overrides the policy's contextual judgment with a dumb "first available corner" rule. This confirms the Plan 563 audit conclusion: blind heuristics cannot improve on a trained policy within the policy's training distribution. Opening books are closed. Full record: `.benchmarks/204_opening_book_vs_moka_negative.md`.

### PUCT search — the AlphaZero recipe (Bench 205, the new best)

The one remaining architectural lever. `GoPuctMokaPlayer` implements PUCT (Predictor + UCB applied to Trees) — the AlphaZero search algorithm — using BOTH of Moka's heads: the **policy head** provides the exploration prior P(s,a), and the **value head** evaluates leaves. This is structurally different from the existing `GoMctsMokaPlayer` (negative result above), which used UCB1 (ignores the policy prior entirely). The PUCT formula:

```
PUCT(s,a) = Q(s,a) + c_puct · P(s,a) · √(N_parent) / (1 + N(s,a))
```

**Result: massive win.** n=100 per arm, all vs Moka greedy, GO_OPENING_MOVES=4:

| Config | Win% | µs/move |
|---|---|---|
| Alpha-beta (depth=1, top_k=4) — prior best | 74.0% | 2,016 |
| PUCT budget=50, c=1.5, top_k=8 | **94.0%** | 21,129 |
| PUCT budget=100, c=1.5, top_k=8 | **96.0%** | 42,936 |
| PUCT budget=200, c=2.5, top_k=8 | **98.0%** | 79,677 |

Win rate scales with budget (94→96→98%), confirming the search adds real value. Even the cheapest PUCT (budget=50, ~21ms/move) jumps from 74% to 94% — z=4.52, p≈10⁻⁶. Baseline 74% vs PUCT-200 98%: z=5.63, p≈10⁻⁸.

**Why PUCT beats alpha-beta:** the policy prior directs exploration toward moves the network rates highly (critical for Go's ~80 branching factor), and MCTS adaptively focuses simulations on promising lines (vs alpha-beta's fixed shallow depth). The existing `GoMctsMokaPlayer` (UCB1 + value head, negative result) lacked the policy prior — PUCT's addition of P(s,a) is what closes the gap.

**Implementation note:** arena-based tree (`Vec<PuctNode>`), softmax-normalized top_k priors, negamax backprop with Q negated in selection. A sign bug (total_value perspective mismatch) was found and fixed during development — pre-fix scored 0W/20L (search chose worst moves), post-fix 94-98%. Full record: `.benchmarks/205_puct_search_vs_moka_win.md`.

## Plan 565 — Real browser side-by-side + wasmi (the size/speed question, answered for real)

Every prior "Moka: ~0.5ms/move" figure in this document was **our own native port acting as its own baseline** — never the actual browser-deployed Moka. This plan closes that gap with real measurements: the real `github.com/millionco/moka` package, built from source and run in real Chrome (via Playwright, driving the actually-installed `Google Chrome.app`, not a downloaded test browser); our own port compiled to `wasm32-unknown-unknown` and run in the same real Chrome; and the same `.wasm` binary run through `wasmi` (pure interpreter, no JIT) natively. New crate: `crates/katgpt-moka-wasm` — a dependency-free reimplementation of the forward pass (no `katgpt-core`, so no `ahash`/`getrandom` wasm32 backend friction a browser build has no reason to carry) plus a from-scratch minimal 9×9 board, kept faithful to `katgpt_pruners::go::moka_net` by direct port, not reimplementation-from-memory.

**Correction along the way, worth stating plainly:** Moka's actual shipped runtime is **100% hand-written TypeScript**, not WASM at all — traced every code path in the cloned repo; the only `WebAssembly` reference anywhere is ONNX Runtime running KataGo (their much bigger teacher, used as an arena opponent), unrelated to Moka itself. The comparison is therefore "our Rust→WASM vs their hand-written JS," both JIT-compiled by V8 — not "WASM vs WASM," which was the wrong initial framing.

### Results (all real measurements — real Chrome via Playwright, Node V8 re-bench, or native wasmi — not self-benchmarked)

| Engine | Median latency/move | Total payload |
|---|---|---|
| **Real Moka** (their actual JS, real Chrome) | **6.4 ms** | 140,850 B (11,004 JS + 16,198 manifest + 113,648 weights) |
| Our WASM, **no SIMD** (default `wasm32-unknown-unknown`, real Chrome) | 8.6 ms — **slower than Moka** | 267,914 B (9,847 JS glue + 258,067 wasm) |
| **Our WASM, `+simd128`** (real Chrome) | **0.6 ms — 10.7× faster than Moka** | 269,405 B (9,847 JS glue + 259,558 wasm) |
| Real Moka JS (Node V8 JIT, re-bench 2026-07-31) | 7.2 ms | same dist |
| **Our WASM `+simd128`** (Node V8 JIT, re-bench 2026-07-31) | **0.59 ms — 12.2× faster than Moka** | same wasm |
| Our WASM via **wasmi** (pure interpreter, no JIT, SIMD-on, one forward pass) | 76 ms | n/a (native test, no browser payload) |
| Native Rust (NEON, no browser at all) | ~0.45–0.54 ms | n/a |

### What actually happened, in order

1. **First WASM attempt lost to plain JS** (8.6 ms vs 6.4 ms) — surprising, and initially looked like WASM just isn't worth it for a model this small (echoing the ANE finding from Issue 564: fixed overhead dominating a tiny workload).
2. **Diagnosis, not resignation:** `wasm32-unknown-unknown` defaults to no SIMD. `katgpt_types::simd::simd_dot_f32`'s wasm32 SIMD path is gated on `target_feature = "simd128"`, which isn't on by default — so the WASM build was silently running the scalar fallback, throwing away the exact vectorization advantage that made the native port fast in the first place (Plan 563's 8.7× kernel work).
3. **`RUSTFLAGS='-C target-feature=+simd128'` + `wasm-opt --enable-simd` fixed it completely**: 8.6 ms → 0.6 ms (14.3×), landing at **10.7× faster than real Moka**, confirmed in the same real browser, same self-play-generated position, same measurement methodology.
4. **wasmi confirms JIT compilation is where nearly all the performance lives**, not the WASM format itself: the identical binary, pure-interpreted, is 76 ms/call (re-bench 2026-07-31; was 212 ms in the original Plan 565 build — the wasm has been optimized since via Issues 204–207) — ~130× slower than the exact same binary JIT'd by V8. A WASM interpreter alone is not viable for this workload at any point on this ladder.

### Honest final scorecard — three tables (the combined build IS now measured)

Strength (native), greedy-browser-speed, and the COMBINED build (PUCT in real Chrome) come from **three different experiments**. An earlier version of this doc split them into two tables with a footnote — "Nothing in table A has been ported to WASM" — which read as defensive framing: it preserved the flattering "10.7× faster" headline while sidestepping the combined measurement that would settle whether combining PUCT+WASM inverts both headlines. Issue 204 closed that gap by actually porting PUCT into `katgpt-moka-wasm` and measuring it in real Chrome. The result is Table C; the footnote is gone.

**A. Native strength — search algorithm comparison (no browser, no WASM, all native Rust)**

| Config | Win% vs Moka | Latency/move (native) | n |
|---|---|---|---|
| Alpha-beta (depth=1, top_k=4) | 70.0% (n=300); **74.0% reconfirmed fresh, n=100, 1,976.8 µs/move** | ~2.0–2.8 ms | 300 |
| PUCT (budget=50, c=1.5, top_k=8) | 94.0% | ~21 ms | 100 |
| PUCT (budget=100, c=1.5, top_k=8) | 96.0% | ~43 ms | 100 |
| PUCT (budget=200, c=2.5, top_k=8) | **98.0% reconfirmed fresh** (98W/2L, n=100) | **81,099.6 µs ≈ 81.1 ms** | 100 |

Both rows above re-run fresh in this session, on demand, to confirm the documented figures weren't stale — both matched (74.0%/1.98ms and 98.0%/81.1ms respectively). PUCT strictly dominates alpha-beta on strength, at a real and substantial latency cost — budget=200 is **~41× slower per move** than alpha-beta (81.1 ms vs 1.98 ms), not a free upgrade.

**B. Browser speed/size — plain greedy network inference only (no search)**

| | Moka (real) | Ours |
|---|---|---|
| Latency/move | 6.4 ms | **0.5 ms** (zero-copy API) / 0.6 ms (marshalled API) |
| Bundle size | 140,850 B | 273,218 B |

This measures one greedy forward pass, analogous to native `MokaPlayer` (argmax over the raw policy), not any search player. This is the build that is 10.7× faster than Moka — but it uses Moka's own ported weights, so greedy-vs-Moka is a mirror match (~50% win rate, no strength advantage either way). The speed win is real; the strength advantage only appears with search on top (Table C).

**C. The combined build — PUCT search ported to WASM, real Chrome (Issue 204 + win-rate follow-up)**

The measurement the prior doc sidestepped. `GoPuctMokaPlayer` (from `katgpt-pruners`) ported into `katgpt-moka-wasm` as `WasmPuctPlayer`, adapted to that crate's standalone `Board` (no `katgpt-core` dep — same forward pass, same vendored weights, same feature encoder, only the board wrapper changed). Both latency and win-rate are now measured:

| Config | Win% vs Moka (WASM-via-wasmi) | n | Median ms/move (Node V8 JIT) | Avg nodes/move | ms/node |
|---|---|---|---|---|---|
| PUCT budget=50, c=1.5, top_k=8 (**f32**) | **100.0%** (20/20) | 20 | **29.6 ms** | 50 | 0.592 |
| PUCT budget=50, c=1.5, top_k=8 (**int8, DEFAULT**) | **85.0%** (17/20) | 20 | **25.8 ms** | 50 | 0.516 |
| PUCT budget=100, c=1.5, top_k=8 (f32) | — (b50 dominates) | — | **59.8 ms** | 100 | 0.598 |
| PUCT budget=200, c=2.5, top_k=8 (f32) | — (b50 dominates) | — | **119.6 ms** | 200 | 0.598 |

**Issue 206/207 (2026-07-31):** the **int8 row is now the DEFAULT** path
(`PuctPlayer::new` / `wasmi_arena_init(..., 1)` / `WasmPuctPlayer::new` all use
int8 — shipped in commit `7da5cf76`; the tracking issue was removed per the
standard noise-reduction rule once resolved). The f32 path is reachable via
`PuctPlayer::with_f32` / `wasmi_arena_init_f32`. The int8 win-rate (85%) is
below f32's 100% but within the n=20 binomial noise band (Wilson 95% CI on
85% at n=20 ≈ 64–95%; on 100% ≈ 83–100%) — both clear the 75% parity floor.
See [Benchmark 565](../../.benchmarks/565_int8_int8_sdot_positive.md).

Win-rate is measured via wasmi (`tests/wasmi_puct_winrate.rs` for f32, `tests/wasmi_puct_int8_winrate.rs` for int8) — a deterministic IEEE-754 interpreter, so the moves chosen (and therefore the win rate) are bit-identical to what Chrome's V8 JIT would produce for the same binary + inputs. Only b50 was run (871s for f32 n=20, 681s for int8 n=20); b100/b200 strictly dominate b50 on strength, so their win rates are bounded below by b50's — they were not re-run. Native Bench 205's b50 was 94% (n=100); the 100% f32 / 85% int8 here are consistent (at p=0.94, P(20/20) ≈ 29% — a normal high draw; the int8 quantization noise costs a few games at small n but is within noise). Native figures for reference: b50=94%, b100=96%, b200=98% (all n=100, Table A).

Latency scales linearly (29.6→59.8→119.6 doubles at each step) — pure per-simulation cost dominates, no fixed overhead amortizing away. Per-node: **~0.59 ms**, of which the forward pass alone is ~0.5 ms (Table B's figure), so tree overhead (board clone, softmax prior, arena push, negamax backprop) is only ~0.09 ms/node — a ~18% tax on the forward pass.

**Latency measured via Node.js V8 JIT** (same engine as Chrome — `node crates/katgpt-moka-wasm/bench/bench_puct.js` loads the raw `.opt.wasm` via `WebAssembly.instantiate` and times the arena C-ABI exports). Earlier Playwright/Chrome numbers (29.8/59.4/118.1) matched within noise; Node V8 is faster to drive (no browser harness setup) and re-runs cleanly on every rebuild. **Rebuilding:** `./scripts/build-moka-wasm.sh` encodes the full pipeline (`RUSTFLAGS='-C target-feature=+simd128'` + `wasm-bindgen --target nodejs` + `wasm-opt -Oz --enable-simd`) — without the SIMD flags the build silently falls back to the scalar dot kernel (~16× slower, ~500 ms/move). The K-sweep variant (`bench/bench_k_sweep.js`) reproduces the Issue 205 diminishing-returns table.

**wasmi upper bound** (pure interpreter, no JIT, SIMD-on): b50=1,260 ms, b100=2,508 ms, b200=5,031 ms per move. ~46× slower than V8 JIT — confirms JIT compilation is where ~98% of the performance lives.

**Tree-allocation optimization (attempted, honest result):** the PUCT tree hot path was rewritten to eliminate ~800 heap allocations/move (`Board` `Vec<Cell>`→`[Cell;81]`+Copy; `neighbors()` `Vec`→stack `[usize;4]`; `would_be_suicide` uses zero-alloc early-exit `has_liberty` instead of full `flood_group`). Under wasmi this gave 7–9% (b100: 2744→2508, b200: 5462→5031). Under V8 JIT: **within noise** (29.6 vs 29.8ms original). The forward pass is 84% of per-node cost (0.5ms × 50 nodes = 25ms); tree allocations are ~0.2% of total. **Tree-side optimization cannot move the needle** — the paths below 30ms turned out to be (a) int8×int8 dots via `i8x16.dot_s`/SDOT [Issue 206/207, DEFAULT-ON, b50=25.8ms] and (b) batched MCTS (evaluate K leaves per forward pass instead of 1, Issue 205 — marginal 1.09× at K=8).

**Batched MCTS (Issue 205, attempted, honest result):** the batched forward pass + virtual loss + leaf queueing was implemented and measured. K=8 gives **1.09× speedup** (33.7→30.8 ms/move at b50) — far below the estimated 3–5×. K=50 (single giant batch) reaches 1.19× (33.7→28.2ms) with diminishing returns. The forward pass is **compute-bound, not cache-bound**: the Moka net (~100KB weights) fits in L2 cache, so sequential passes already benefit from cache residency, and the SIMD dot kernel already saturates the 128-bit FPU per call. Batching K samples through the same weight slice doesn't reduce total FLOPs. The batched code stays opt-in via `PuctPlayer::with_batch_k(budget, c_puct, top_k, batch_k)`; K=1 (sequential) remains the default (preserves wasmi parity). Full analysis in [Benchmark 205](../../.benchmarks/205_puct_wasm_batched_mcts_latency.md).

**Correction (Issue 206/207, 2026-07-31):** the "30ms floor is the real Moka-net-on-CPU floor" conclusion below was **wrong** — it held only for the f32 path. The int8×int8 forward path (Bench 565) broke the floor: PUCT b50 dropped to **25.8 ms** under V8 JIT (1.17–1.19× over f32's 30.6 ms) by routing the dot kernel through `i8x16.dot_s` (WASM SIMD128) / SDOT (aarch64 dotprod) — different execution units than the f32 FPU, which the original "FPU saturated" finding (Bench 205) was about. The int8 path is now **DEFAULT-ON** (commit `7da5cf76`, 2026-07-31 — win-rate parity gate cleared at 85% vs f32's 100% at n=20, both above the 75% floor; the tracking issue was removed per the standard noise-reduction rule once resolved). See [Benchmark 565](../../.benchmarks/565_int8_int8_sdot_positive.md). The original "not pursued" conclusion on im2col/GEMM/WebGPU/smaller-net stands — those remain the only paths to *dramatic* (≥2×) further improvement.

**Summary matrix (each build's trade-off, one row each):**

| Build | Win% vs Moka | Latency/move | Bundle size | Faster than Moka? | Stronger than Moka? |
|---|---|---|---|---|---|
| Moka (real, reference) | — (baseline) | 6.4 ms | 140,850 B | — | — |
| Ours — greedy (Table B) | ≈50% (mirror — same weights) | 0.5 ms | 273,218 B | **yes (10.7×)** | no |
| Ours — PUCT b50, **int8 (DEFAULT since Issue 207)** | **85%** (WASM, n=20) / 94% (native, n=100) | **25.8 ms** (WASM V8) | 273,218 B | no (~4.0× slower) | **yes** |
| Ours — PUCT b50, f32 (`with_f32` escape hatch) | **100%** (WASM, n=20) / 94% (native, n=100) | 29.6 ms (WASM V8) / ~21 ms (native) | 273,218 B | no (~4.6× slower) | **yes** |
| Ours — PUCT b200, f32 | 98% (native, n=100) | 119.6 ms (WASM V8) / ~81 ms (native) | 273,218 B | no (~18.7× slower) | **yes** |

No single build is both faster AND stronger than Moka. The greedy build wins on speed but loses on strength; the PUCT builds win on strength but lose on speed. These are the measured numbers — not projected.

**Win-rate source note:** the b50 figures (f32 100%, int8 85%) are WASM-via-wasmi (n=20 each); the b200 98% is native (n=100). These are different experiments at different sample sizes — NOT a sign that WASM/f32 is stronger than native/int8. Native and WASM produce **identical moves** for a given position *within the same dtype* (same weights, same forward pass, deterministic IEEE-754 execution), so their true win rates are identical. The 100% (f32 WASM, n=20) vs 94% (f32 native, n=100) gap at b50 is sampling variance: at true p=0.94, P(20/20) ≈ 29% — a normal high draw. The int8 85% vs f32 100% gap is the quantization noise — int8 introduces ~2% relative logit error, which occasionally flips a move choice and costs a game at small n, but stays within the binomial noise band (Wilson 95% CI on 85% at n=20 ≈ 64–95%). Both clear the 75% parity floor decisively. The WASM measurements used n=20 (not 100) because wasmi is ~46× slower than V8 JIT, so 100 games would take ~73 minutes vs ~14.5 minutes.

### Follow-up: does zero-copy JS↔wasm sharing help further?

`WasmMoka::infer(&[f32]) -> Vec<f32>` marshals on both sides of every call: the input slice is bulk-copied into a temporary wasm buffer, and the `Vec<f32>` return is allocated in wasm, copied out to a fresh JS array, then freed. WASM linear memory is inherently JS-visible zero-copy (`WebAssembly.Memory.buffer` wrapped directly in a `Float32Array` view, no `SharedArrayBuffer`/threads needed) — so `WasmMoka::infer_ptr`/`WasmGame::encode_features_ptr` were added: persistent, never-resized buffers with stable addresses, exposed via `wasm_bindgen::memory()`, read/written through JS-side `Float32Array` views instead of marshalled arguments/return values.

**Paired same-page-load result (n=100 each, real Chrome):**

| | Median | Mean |
|---|---|---|
| Baseline (`infer`, marshalled) | 0.60 ms | 0.55 ms |
| Zero-copy (`infer_ptr`, shared memory view) | 0.50 ms | 0.54 ms |

**Real, measurable, but modest** — roughly 15–20% at the median, with p99/min/max identical between the two. Marshalling cost is real, but it was never the dominant cost once SIMD was already enabled; the 14.3× win from Phase 1 (enabling `simd128`) dwarfs this. This widens an already-decisive lead rather than fixing a loss. Size cost of keeping both APIs side-by-side for the A/B: +3,813 B (269,405 → 273,218) — a real deployment would drop the marshalled `infer`/`encode_features` entirely rather than keep both, likely netting smaller, not larger.