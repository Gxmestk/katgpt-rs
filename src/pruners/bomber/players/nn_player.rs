//! P3b: NNPlayer — heuristic scoring + WASM safety validation.
//!
//! Extracted from `players.rs` (Issue 175).

use std::any::Any;

use fastrand::Rng;

use crate::pruners::bomber::wasm_pruner::BomberWasmPruner;
use crate::pruners::bomber::{ArenaGrid, BomberAction, GameEvent, GridPos};

use super::helpers::{
    action_index, escape_distance, in_blast_zone, is_safe_action, move_target, score_action,
    update_bombs, update_powerups,
};
use super::{ALL_ACTIONS, BOMB_FUSE_TICKS, BomberPlayer, DEFAULT_BLAST_RANGE, KnownBomb};

/// P2.5: Neural Network + WASM Validator — heuristic scoring + WASM safety.
///
/// Combines heuristic action scoring (same as ValidatorPlayer) with
/// WASM-based safety validation. If `bomber_validator.wasm` is loaded,
/// safety checks run in the WASM sandbox. Falls back to native Rust
/// safety rules if WASM is unavailable.
///
/// Future: LoRA model proposals (Phase 1, blocked by training corpus).
pub struct NNPlayer {
    pub(crate) _id: u8,
    pub(crate) wasm: Option<BomberWasmPruner>,
    pub(crate) known_bombs: Vec<KnownBomb>,
    pub(crate) known_powerups: Vec<(i32, i32)>,
    pub(crate) last_dir: Option<BomberAction>,
}

impl NNPlayer {
    /// Create NNPlayer with WASM validator loaded from file.
    ///
    /// Falls back to native safety rules if WASM fails to load.
    pub fn new_with_wasm(id: u8, wasm_path: &str) -> Self {
        let wasm = BomberWasmPruner::load_from_file(wasm_path).ok();
        Self {
            _id: id,
            wasm,
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            last_dir: None,
        }
    }

    /// Create NNPlayer without WASM (native fallback only).
    pub fn new_native(id: u8) -> Self {
        Self {
            _id: id,
            wasm: None,
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            last_dir: None,
        }
    }

    /// Check if action is safe — WASM if available, native otherwise.
    fn is_action_safe(
        &self,
        action: &BomberAction,
        grid: &ArenaGrid,
        pos: GridPos,
        bombs: &[KnownBomb],
    ) -> bool {
        if let Some(ref wasm) = self.wasm {
            return wasm.is_safe_action(action_index(action), grid, pos.x, pos.y, self._id, bombs);
        }
        is_safe_action(action, grid, pos, bombs)
    }
}

impl BomberPlayer for NNPlayer {
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
                        None => -5.0,
                    };
                if score > best_score {
                    best_score = score;
                    best = *action;
                }
            } else {
                // Safe mode: hard-block unsafe actions via WASM or native
                if !self.is_action_safe(action, grid, pos, &self.known_bombs) {
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
        match self.wasm {
            Some(_) => "NN-WASM",
            None => "NN-Native",
        }
    }

    fn emoji(&self) -> &str {
        "🤖"
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
