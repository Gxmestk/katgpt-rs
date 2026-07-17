//! P3: ValidatorPlayer — policy scoring with safety validation.
//!
//! Extracted from `players.rs` (Issue 175).

use std::any::Any;

use fastrand::Rng;

use super::{
    ALL_ACTIONS, BOMB_FUSE_TICKS, DEFAULT_BLAST_RANGE, ArenaGrid, BomberAction, BomberPlayer,
    GameEvent, GridPos, KnownBomb,
};
use super::helpers::{
    escape_distance, in_blast_zone, is_safe_action, move_target, score_action, update_bombs,
    update_powerups,
};

/// P3: Model + Validator — policy scoring with safety validation.
///
/// Same policy scoring as P2 but adds a hard safety filter:
/// - Only considers actions that pass `is_safe_action`
/// - Never walks into active blast zones, walls, or places bomb without escape
pub struct ValidatorPlayer {
    pub(crate) _id: u8,
    pub(crate) known_bombs: Vec<KnownBomb>,
    pub(crate) known_powerups: Vec<(i32, i32)>,
    pub(crate) last_dir: Option<BomberAction>,
}

impl ValidatorPlayer {
    pub fn new(id: u8) -> Self {
        Self {
            _id: id,
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            last_dir: None,
        }
    }
}

impl BomberPlayer for ValidatorPlayer {
    fn select_action(
        &mut self,
        grid: &ArenaGrid,
        pos: GridPos,
        events: &[GameEvent],
        _rng: &mut Rng,
    ) -> BomberAction {
        update_bombs(&mut self.known_bombs, events);
        update_powerups(&mut self.known_powerups, events);

        let in_danger = in_blast_zone(pos, grid, &self.known_bombs);
        // O(bombs) linear helper — replaces per-call HashSet allocation.
        let is_blocked = |x: i32, y: i32| {
            self.known_bombs
                .iter()
                .any(|(p, _, _)| p.0 == x && p.1 == y)
        };

        let mut best = BomberAction::Wait;
        let mut best_score = f32::NEG_INFINITY;

        for action in &ALL_ACTIONS {
            let is_move = matches!(
                action,
                BomberAction::Up | BomberAction::Down | BomberAction::Left | BomberAction::Right
            );

            if in_danger {
                // Escape mode: score movement by escape distance, skip Bomb/Wait
                if !is_move {
                    continue;
                }
                let target = move_target(action, pos);
                if !grid.is_walkable(target.x, target.y) || is_blocked(target.x, target.y) {
                    continue;
                }
                let score =
                    match escape_distance(target, grid, &self.known_bombs, &self.known_bombs) {
                        Some(dist) => 10.0 - dist as f32 * 0.5,
                        None => -5.0, // No escape route found — try anyway
                    };
                if score > best_score {
                    best_score = score;
                    best = *action;
                }
            } else {
                // Safe mode: hard-block unsafe actions (validator's purpose)
                if !is_safe_action(action, grid, pos, &self.known_bombs) {
                    continue;
                }
                // Detonate validation: only valid when active bombs exist and safe to detonate.
                // Future: restrict to Remote bombs only once bomb_type is tracked in KnownBomb.
                if *action == BomberAction::Detonate
                    && (self.known_bombs.is_empty() || in_blast_zone(pos, grid, &self.known_bombs))
                {
                    continue;
                }
                let score = score_action(
                    action,
                    grid,
                    pos,
                    &self.known_bombs,
                    &self.known_powerups,
                    self.last_dir,
                );
                if score > best_score {
                    best_score = score;
                    best = *action;
                }
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
        "Validator"
    }

    fn emoji(&self) -> &str {
        "🐶"
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
