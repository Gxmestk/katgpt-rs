//! P2b: LoraPlayer — trained LoRA model for action scoring.
//!
//! Extracted from `players.rs` (Issue 175).

use std::any::Any;

use fastrand::Rng;

use crate::pruners::bomber::{ArenaGrid, BomberAction, GameEvent, GridPos};
use crate::types::LoraAdapter;

use super::{ALL_ACTIONS, BOMB_FUSE_TICKS, DEFAULT_BLAST_RANGE, BomberPlayer, KnownBomb};
use super::helpers::{
    lora_score_actions, move_target, score_action, update_bombs, update_powerups,
};

/// P2b: LoRA-only player — uses trained LoRA model for action scoring.
///
/// No WASM validator, no bandit. Proves LoRA > random.
/// Falls back to heuristic scoring if LoRA fails to load or apply.
pub struct LoraPlayer {
    pub(crate) _id: u8,
    pub(crate) lora: Option<LoraAdapter>,
    pub(crate) lora_buf: Vec<f32>,
    pub(crate) known_bombs: Vec<KnownBomb>,
    pub(crate) known_powerups: Vec<(i32, i32)>,
    pub(crate) last_dir: Option<BomberAction>,
}

impl LoraPlayer {
    /// Create LoraPlayer without LoRA (heuristic fallback).
    pub fn new(id: u8) -> Self {
        Self {
            _id: id,
            lora: None,
            lora_buf: Vec::new(),
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            last_dir: None,
        }
    }

    /// Create LoraPlayer with LoRA loaded from file.
    ///
    /// Only loads the first adapter — multi-adapter L2+ files have layers 1+
    /// silently dropped. For full multi-adapter evaluation, switch to a player
    /// that applies each adapter to its target projection during forward pass.
    pub fn new_with_lora(id: u8, lora_path: &str) -> Self {
        let lora = LoraAdapter::load_first(std::path::Path::new(lora_path)).ok();
        let buf_size = lora.as_ref().map_or(0, |l| l.rank);
        Self {
            _id: id,
            lora,
            lora_buf: vec![0.0; buf_size],
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            last_dir: None,
        }
    }
}

impl BomberPlayer for LoraPlayer {
    fn select_action(
        &mut self,
        grid: &ArenaGrid,
        pos: GridPos,
        events: &[GameEvent],
        rng: &mut Rng,
    ) -> BomberAction {
        update_bombs(&mut self.known_bombs, events);
        update_powerups(&mut self.known_powerups, events);

        // O(bombs) linear helper — replaces per-call HashSet allocation.
        let is_blocked = |x: i32, y: i32| {
            self.known_bombs
                .iter()
                .any(|(p, _, _)| p.0 == x && p.1 == y)
        };

        // Try LoRA scoring first
        let scores = self.lora.as_ref().and_then(|lora| {
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

            // Basic wall collision filter
            if is_move {
                let target = move_target(action, pos);
                if !grid.is_walkable(target.x, target.y) || is_blocked(target.x, target.y) {
                    continue;
                }
            }

            let score = match &scores {
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

        // 10% random exploration (epsilon-greedy)
        if rng.f32() < 0.10 {
            let safe_moves: Vec<BomberAction> = ALL_ACTIONS
                .iter()
                .filter(|a| {
                    if matches!(
                        a,
                        BomberAction::Up
                            | BomberAction::Down
                            | BomberAction::Left
                            | BomberAction::Right
                    ) {
                        let target = move_target(a, pos);
                        grid.is_walkable(target.x, target.y)
                    } else {
                        false
                    }
                })
                .copied()
                .collect();
            if !safe_moves.is_empty() {
                best = safe_moves[rng.usize(0..safe_moves.len())];
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
        match self.lora {
            Some(_) => "LoRA",
            None => "LoRA-Fallback",
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
