//! P2: GreedyPlayer — policy scoring with exploration.
//!
//! Extracted from `players.rs` (Issue 175).

use std::any::Any;

use fastrand::Rng;

use super::helpers::{in_blast_zone, move_target, score_action, update_bombs, update_powerups};
use super::{
    ALL_ACTIONS, ArenaGrid, BOMB_FUSE_TICKS, BomberAction, BomberPlayer, DEFAULT_BLAST_RANGE,
    GameEvent, GridPos, KnownBomb,
};

/// P2: Model-based — policy scoring with exploration.
///
/// Scores all actions using clear policy priorities (flee > bomb > hunt > explore)
/// and picks the best. Adds 20% random exploration to discover new strategies.
pub struct GreedyPlayer {
    pub(crate) _id: u8,
    pub(crate) known_bombs: Vec<KnownBomb>,
    pub(crate) known_powerups: Vec<(i32, i32)>,
    pub(crate) last_dir: Option<BomberAction>,
}

impl GreedyPlayer {
    pub fn new(id: u8) -> Self {
        Self {
            _id: id,
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            last_dir: None,
        }
    }
}

impl BomberPlayer for GreedyPlayer {
    fn select_action(
        &mut self,
        grid: &ArenaGrid,
        pos: GridPos,
        events: &[GameEvent],
        rng: &mut Rng,
    ) -> BomberAction {
        update_bombs(&mut self.known_bombs, events);
        update_powerups(&mut self.known_powerups, events);

        // 20% random exploration — only safe movement, never random bomb
        if rng.f32() < 0.2 {
            let safe_moves: Vec<BomberAction> = [
                BomberAction::Up,
                BomberAction::Down,
                BomberAction::Left,
                BomberAction::Right,
            ]
            .into_iter()
            .filter(|&action| {
                let target = move_target(&action, pos);
                grid.is_walkable(target.x, target.y)
                    && !in_blast_zone(target, grid, &self.known_bombs)
            })
            .collect();
            if !safe_moves.is_empty() {
                let action = safe_moves[rng.usize(0..safe_moves.len())];
                self.last_dir = Some(action);
                return action;
            }
        }

        // Policy: score all actions, pick best.
        // Pre-compute once: the max_by closure would otherwise call score_action
        // ~2×(N-1) times (each invocation recomputes the bomb_positions HashSet
        // and runs escape_distance BFS).
        let mut best = BomberAction::Wait;
        let mut best_score = f32::NEG_INFINITY;
        for &action in &ALL_ACTIONS {
            let s = score_action(
                &action,
                grid,
                pos,
                &self.known_bombs,
                &self.known_powerups,
                self.last_dir,
            );
            if s > best_score {
                best_score = s;
                best = action;
            }
        }

        if matches!(
            best,
            BomberAction::Up | BomberAction::Down | BomberAction::Left | BomberAction::Right
        ) {
            self.last_dir = Some(best);
        }
        if best == BomberAction::Bomb {
            self.known_bombs
                .push(((pos.x, pos.y), DEFAULT_BLAST_RANGE, BOMB_FUSE_TICKS));
        }
        best
    }

    fn name(&self) -> &str {
        "Greedy"
    }

    fn emoji(&self) -> &str {
        "🐱"
    }

    fn reset(&mut self) {
        self.known_bombs.clear();
        self.known_powerups.clear();
        self.last_dir = None;
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
