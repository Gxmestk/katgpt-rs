//! AI player trait and implementations for Bomberman HL Arena.
//!
//! Player types representing increasing HL technology levels:
//! - P1 (Random): no model, no learning — pure baseline
//! - P2 (Greedy): heuristic action selection
//! - P2b (LoraPlayer): trained LoRA model scoring — proves LoRA > random
//! - P3 (Validator): heuristic + hard safety rules
//! - P3b (NNPlayer/WasmPlayer): WASM validator sandbox — proves safety > none
//! - P4 (LoraWasmPlayer): LoRA proposals + WASM validation — proves synergy
//! - P5 (HLPlayer): LoRA + WASM + Bandit + AbsorbCompress — proves adaptation
//!
//! Issue 175: previously a single 2828-line `players.rs`, now a directory
//! with one file per player type + `helpers.rs` (shared utilities) + `tests.rs`.

// Bring bomber primitives into scope for use in the trait definition + factory.
// (Submodules import directly from `crate::pruners::bomber::...`.)
use crate::pruners::bomber::{ArenaGrid, BomberAction, GameEvent, GridPos};

// `Rng` and `Any` are used in the `BomberPlayer` trait definition.
use fastrand::Rng;
use std::any::Any;

// ── Constants ──────────────────────────────────────────────────

pub(crate) const ACTION_COUNT: usize = 7;
pub(crate) const DEFAULT_BLAST_RANGE: u32 = 2;
pub(crate) const BOMB_FUSE_TICKS: u32 = crate::pruners::bomber::BOMB_FUSE_TICKS;

pub(crate) const ALL_ACTIONS: [BomberAction; ACTION_COUNT] = [
    BomberAction::Up,
    BomberAction::Down,
    BomberAction::Left,
    BomberAction::Right,
    BomberAction::Bomb,
    BomberAction::Wait,
    BomberAction::Detonate,
];

/// Tracked bomb: (position, blast_range, fuse_ticks_remaining).
pub(crate) type KnownBomb = ((i32, i32), u32, u32);

/// Tracked opponent: (player_id, current_pos, prev_pos).
pub(crate) type KnownOpponent = (u8, (i32, i32), Option<(i32, i32)>);

// ── Trait ──────────────────────────────────────────────────────

/// AI player trait for Bomberman arena.
///
/// Each implementation represents a different HL technology level:
/// - P1 (Random): no model, no learning
/// - P2 (Model): LoRA-based action selection
/// - P3 (Validated): LoRA + WASM validator
/// - P4 (Full HL): LoRA + WASM + Bandit + TrialLog + AbsorbCompress
pub trait BomberPlayer {
    /// Select an action given the current game state.
    fn select_action(
        &mut self,
        grid: &ArenaGrid,
        pos: GridPos,
        events: &[GameEvent],
        rng: &mut Rng,
    ) -> BomberAction;

    /// Player display name.
    fn name(&self) -> &str;

    /// Emoji for TUI rendering.
    fn emoji(&self) -> &str;

    /// Reset internal state for a new round.
    fn reset(&mut self);

    /// Downcast support for HL player updates.
    fn as_any(&self) -> &dyn Any;

    /// Downcast support for HL player updates (mutable).
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

// ── Submodules ─────────────────────────────────────────────────

mod greedy_player;
mod helpers;
mod hl_player;
mod random_player;
mod validator_player;

#[cfg(feature = "bomber-wasm")]
mod lora_player;
#[cfg(feature = "bomber-wasm")]
mod lora_wasm_player;
#[cfg(feature = "bomber-wasm")]
mod nn_player;

#[cfg(test)]
mod tests;

// ── Re-exports (external API surface) ──────────────────────────

pub use greedy_player::GreedyPlayer;
pub use hl_player::HLPlayer;
pub use random_player::RandomPlayer;
pub use validator_player::ValidatorPlayer;

#[cfg(feature = "bomber-wasm")]
pub use lora_player::LoraPlayer;
#[cfg(feature = "bomber-wasm")]
pub use lora_wasm_player::LoraWasmPlayer;
#[cfg(feature = "bomber-wasm")]
pub use nn_player::NNPlayer;

// Public helpers consumed by external modules
// (`replay_backward.rs`, `gate_player.rs`, `sonlt_player.rs`, `tft_player.rs`,
// `validator_agent.rs`, `blend_context.rs`).
pub use helpers::is_safe_action;

// `pub(crate)` helpers consumed by 9+ sibling player modules
// (`sdpg_player.rs`, `rmsd_player.rs`, `validator_agent.rs`, `gate_player.rs`,
// `sonlt_player.rs`, `tft_player.rs`, `g_zero_player.rs`, `blend_context.rs`,
// `blend_estimators.rs`, `contextual_bandit.rs`).
pub(crate) use helpers::{
    count_escape_routes, in_blast_zone, intercept_score, is_in_single_blast, move_target,
    predict_direction, score_action, should_place_bomb, trap_score, update_bombs, update_powerups,
};

// ── Factory ────────────────────────────────────────────────────

/// Create the 4 player instances for a tournament.
pub fn create_players() -> Vec<Box<dyn BomberPlayer>> {
    vec![
        Box::new(RandomPlayer::new(0)),
        Box::new(GreedyPlayer::new(1)),
        Box::new(ValidatorPlayer::new(2)),
        Box::new(HLPlayer::new(3)),
    ]
}

/// Create 4 players with NNPlayer (P2.5) replacing ValidatorPlayer (P3).
///
/// If `wasm_path` is `Some`, NNPlayer loads the WASM validator for sandboxed
/// safety checks. Otherwise, uses native Rust safety rules.
#[cfg(feature = "bomber-wasm")]
pub fn create_players_with_wasm(wasm_path: Option<&str>) -> Vec<Box<dyn BomberPlayer>> {
    let p2 = match wasm_path {
        Some(path) => Box::new(NNPlayer::new_with_wasm(2, path)) as Box<dyn BomberPlayer>,
        None => Box::new(NNPlayer::new_native(2)),
    };
    vec![
        Box::new(RandomPlayer::new(0)),
        Box::new(GreedyPlayer::new(1)),
        p2,
        Box::new(HLPlayer::new(3)),
    ]
}
