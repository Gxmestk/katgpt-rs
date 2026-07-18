# Plan 194: Go SDPG Tournament — Self-Distilled Policy Gradient on Go Categories

> **Status:** ✅ Complete — GOAT PASSED (SDPG 56% > HL 44%)
> **Branch:** `develop/feature/194_go_sdpg_tournament`
> **Depends on:** Plan 180 (SDPG Bandit ✅), Plan 065 (Go ✅), Plan 074 (Go HL Credit ✅)
> **Research:** `.research/160_SDPG_Self_Distilled_Policy_Gradient.md`
> **Feature gate:** `sdpg_bandit`, `go`
> **Goal:** Apply SDPG bandit to Go's 8 move categories where Q-values meaningfully differentiate (unlike Bomber's interchangeable templates). Burn-in GoHLPlayer → extract teacher Q → SDPG advantage signal → GOAT gate: SDPG > HL > Greedy > Random.

## Why Go (Not Bomber)

Bomber SDPG (Plan 180) was a **negative result** because all 8 templates converged to Q~0.88 (variance <0.04). Template selection doesn't determine bomber outcomes — bomb timing and safety filters do.

Go's 8 `GoMoveCategory` arms **actually differentiate**:
- **Capture** (arm 3): Direct material gain — strong win correlation
- **CornerStar** (arm 0): Opening theory — strong early-game signal
- **Defend** (arm 4): Defensive necessity — high win/loss contrast
- **Pass** (arm 7): Often endgame resignation — low Q expected
- **Influence** (arm 6): Long-term potential — moderate signal

This gives SDPG's sigmoid advantage real signal to work with: `σ(teacher_Q[Capture]/τ) - σ(student_Q[Capture]/τ)` will be meaningfully positive when the teacher knows Capture is good.

## Architecture

```
GoSdpgPlayer
├── inner: GoHLPlayer (heuristic 80% + bandit 20%)
├── sdpg_bandit: SdpgBanditPruner<NoScreeningPruner> (8 arms = 8 categories)
├── teacher_q: [f32; 8] (from burn-in oracle)
└── category_trace: Vec<(GoMoveCategory, f32)> (per-move delta)
```

Key difference from Bomber: No template proposer layer. Go's categories ARE the SDPG arms directly. The `GoHLPlayer` already categorizes every move — SDPG adds the teacher-student advantage on top.

## Tasks

### Phase 1: GoSdpgPlayer Implementation

- [x] **T1: Create `crates/katgpt-pruners/src/go/sdpg_player.rs`** — `GoSdpgPlayer` struct
  - Wraps `GoHLPlayer` for move categorization and heuristic scoring
  - Own `SdpgBanditPruner<NoScreeningPruner>` with 8 arms (one per `GoMoveCategory`)
  - `with_teacher_q(teacher_q: Vec<f32>)` constructor for oracle injection
  - `update_outcome(won: bool)` feeds category trace to SDPG bandit
  - `GoPlayer` trait impl: blends heuristic + bandit Q + SDPG advantage
  - `as_any_mut()` for downcast in tournament

- [x] **T2: Wire `GoSdpgPlayer` into Go module exports**
  - Add `pub mod sdpg_player;` behind `#[cfg(feature = "sdpg_bandit")]` in `go/mod.rs`
  - Re-export `GoSdpgPlayer`

- [x] **T3: Add `GoPlayerType::Sdpg` variant** (optional, for tournament config)
  - Add to `GoPlayerType` enum in `tournament.rs`
  - Add `create_player()` arm

### Phase 2: Tournament Example

- [x] **T4: Create `examples/go_10_sdpg_tournament.rs`** — Internal round-robin tournament
  - Phase 1: **Burn-in** — Run `GoHLPlayer` vs `GoGreedyPlayer` for N games → extract category Q-values as teacher oracle
  - Phase 2: **SDPG Tournament** — `GoSdpgPlayer(oracle)` vs `HL` vs `Greedy` vs `Random`
  - Phase 3: **GOAT Gate** — Verify SDPG win rate > HL win rate
  - Feature gates: `go,sdpg_bandit`

- [x] **T5: Register example in `Cargo.toml`** with required-features

### Phase 3: Validation

- [x] **T6: Run tournament** — GOAT PASSED: SDPG 56% > HL 44% on 9×9 (200 burn-in, 100 GOAT games)
- [x] **T7: Update `.benchmarks/`** with results

## Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| SDPG win rate | > HL win rate | GOAT gate (Phase 3) |
| Category Q differentiation | variance > 0.1 after burn-in | Teacher Q inspection |
| Sigmoid advantage signal | non-zero for >4 categories | Advantage vector inspection |
| No regression | Existing Go tests still pass | `cargo test --features go,sdpg_bandit` |

## Files Modified

| File | Changes |
|------|---------|
| `crates/katgpt-pruners/src/go/sdpg_player.rs` | **New:** `GoSdpgPlayer` struct + `GoPlayer` impl |
| `crates/katgpt-pruners/src/go/mod.rs` | Add `sdpg_player` module + re-export |
| `crates/katgpt-pruners/src/go/tournament.rs` | Add `GoPlayerType::Sdpg` variant |
| `examples/go_10_sdpg_tournament.rs` | **New:** Burn-in + SDPG tournament + GOAT gate |
| `Cargo.toml` | Register example |
