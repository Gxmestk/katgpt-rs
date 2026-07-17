//! P1: RandomPlayer — uniform random action selection.
//!
//! Extracted from `players.rs` (Issue 175).

use std::any::Any;

use fastrand::Rng;

use super::{
    ALL_ACTIONS, ArenaGrid, BomberAction, BomberPlayer, GameEvent, GridPos, move_target,
};

/// P1: Modelless baseline — uniform random action selection.
///
/// No learning. No memory. No model. Pure baseline.
/// Avoids walking into walls (up to 3 re-rolls, then Wait).
pub struct RandomPlayer {
    pub(crate) _id: u8,
}

impl RandomPlayer {
    pub fn new(id: u8) -> Self {
        Self { _id: id }
    }
}

impl BomberPlayer for RandomPlayer {
    fn select_action(
        &mut self,
        grid: &ArenaGrid,
        pos: GridPos,
        _events: &[GameEvent],
        rng: &mut Rng,
    ) -> BomberAction {
        // Truly random baseline: pick any action with equal probability.
        // Only avoids walking into walls (up to 3 re-rolls, then Wait).
        // No blast zone avoidance, no bomb intelligence.
        for _ in 0..4 {
            let action = ALL_ACTIONS[rng.usize(0..ALL_ACTIONS.len())];
            let target = move_target(&action, pos);
            if matches!(
                action,
                BomberAction::Up | BomberAction::Down | BomberAction::Left | BomberAction::Right
            ) {
                if grid.is_walkable(target.x, target.y) {
                    return action;
                }
            } else {
                return action; // Bomb/Wait/Detonate — always valid
            }
        }
        BomberAction::Wait // All re-rolls hit walls
    }

    fn name(&self) -> &str {
        "Random"
    }

    fn emoji(&self) -> &str {
        "🐰"
    }

    fn reset(&mut self) {}

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
