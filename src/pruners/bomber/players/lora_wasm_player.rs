//! P4: LoraWasmPlayer — LoRA proposals + WASM validation (the synergy player).
//!
//! Extracted from `players.rs` (Issue 175).

use std::any::Any;

use fastrand::Rng;

use crate::pruners::bomber::wasm_pruner::BomberWasmPruner;
use crate::pruners::bomber::{ArenaGrid, BomberAction, GameEvent, GridPos};
use crate::types::LoraAdapter;

use super::helpers::{
    action_index, escape_distance, in_blast_zone, is_safe_action, lora_score_actions, move_target,
    score_action, update_bombs, update_powerups,
};
use super::{ALL_ACTIONS, BOMB_FUSE_TICKS, BomberPlayer, DEFAULT_BLAST_RANGE, KnownBomb};

/// P4: LoRA proposals + WASM validation — the synergy player.
///
/// Model proposes action scores via LoRA, WASM validator filters unsafe ones.
/// Proves LoRA+WASM synergy > either alone.
pub struct LoraWasmPlayer {
    pub(crate) _id: u8,
    pub(crate) lora: Option<LoraAdapter>,
    pub(crate) wasm: Option<BomberWasmPruner>,
    pub(crate) lora_buf: Vec<f32>,
    pub(crate) known_bombs: Vec<KnownBomb>,
    pub(crate) known_powerups: Vec<(i32, i32)>,
    pub(crate) last_dir: Option<BomberAction>,
}

impl LoraWasmPlayer {
    /// Create with no artifacts (heuristic + native safety).
    pub fn new(id: u8) -> Self {
        Self {
            _id: id,
            lora: None,
            wasm: None,
            lora_buf: Vec::new(),
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            last_dir: None,
        }
    }

    /// Create with LoRA only.
    ///
    /// Only loads the first adapter — multi-adapter L2+ files have layers 1+
    /// silently dropped. See `LoraAdapter::load_first` for the limitation.
    pub fn new_with_lora(id: u8, lora_path: &str) -> Self {
        let lora = LoraAdapter::load_first(std::path::Path::new(lora_path)).ok();
        let buf_size = lora.as_ref().map_or(0, |l| l.rank);
        Self {
            _id: id,
            lora,
            wasm: None,
            lora_buf: vec![0.0; buf_size],
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            last_dir: None,
        }
    }

    /// Create with WASM only.
    pub fn new_with_wasm(id: u8, wasm_path: &str) -> Self {
        let wasm = BomberWasmPruner::load_from_file(wasm_path).ok();
        Self {
            _id: id,
            lora: None,
            wasm,
            lora_buf: Vec::new(),
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            last_dir: None,
        }
    }

    /// Create with both artifacts (full LoRA + WASM stack).
    ///
    /// Only loads the first LoRA adapter — multi-adapter L2+ files have layers 1+
    /// silently dropped. See `LoraAdapter::load_first` for the limitation.
    pub fn new_with_secrets(id: u8, lora_path: &str, wasm_path: &str) -> Self {
        let lora = LoraAdapter::load_first(std::path::Path::new(lora_path)).ok();
        let wasm = BomberWasmPruner::load_from_file(wasm_path).ok();
        let buf_size = lora.as_ref().map_or(0, |l| l.rank);
        Self {
            _id: id,
            lora,
            wasm,
            lora_buf: vec![0.0; buf_size],
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

impl BomberPlayer for LoraWasmPlayer {
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

        // Try LoRA scoring
        let lora_scores = self.lora.as_ref().and_then(|lora| {
            lora_score_actions(
                lora,
                grid,
                pos,
                &self.known_bombs,
                &self.known_powerups,
                self.last_dir,
                &mut self.lora_buf,
            )
        });

        let mut best = BomberAction::Wait;
        let mut best_score = f32::NEG_INFINITY;

        for (i, action) in ALL_ACTIONS.iter().enumerate() {
            let is_move = matches!(
                action,
                BomberAction::Up | BomberAction::Down | BomberAction::Left | BomberAction::Right
            );

            if in_danger {
                // Escape mode: skip Bomb/Wait, find escape route
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

                // Use LoRA scores if available, else heuristic
                let score = match &lora_scores {
                    Some(s) => s[i],
                    None => score_action(
                        action,
                        grid,
                        pos,
                        &self.known_bombs,
                        &self.known_powerups,
                        self.last_dir,
                    ),
                };

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
        match (&self.lora, &self.wasm) {
            (Some(_), Some(_)) => "LoRA+WASM",
            (Some(_), None) => "LoRA+Native",
            (None, Some(_)) => "Heuristic+WASM",
            (None, None) => "Heuristic+Native",
        }
    }

    fn emoji(&self) -> &str {
        "🔮"
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
