//! Tests for the bomber players module.
//!
//! Extracted from `players.rs` (Issue 175).

use super::*;
use super::helpers::{action_index, move_target};
use crate::pruners::bomber::{ArenaGrid, BomberAction, GameEvent, GridPos, BOMB_FUSE_TICKS};
use fastrand::Rng;

fn empty_grid() -> ArenaGrid {
    ArenaGrid::generate(42)
}

#[test]
fn test_random_player_valid_actions() {
    let mut player = RandomPlayer::new(0);
    let grid = empty_grid();
    let mut rng = Rng::with_seed(42);
    let pos = GridPos { x: 1, y: 1 }; // Spawn position — always walkable

    for _ in 0..50 {
        let action = player.select_action(&grid, pos, &[], &mut rng);
        // Should never walk into a wall
        if action != BomberAction::Bomb && action != BomberAction::Wait {
            let target = move_target(&action, pos);
            assert!(
                grid.is_walkable(target.x, target.y),
                "RandomPlayer walked into wall at ({},{})",
                target.x,
                target.y,
            );
        }
    }
}

#[test]
fn test_greedy_player_prefers_safety() {
    let mut player = GreedyPlayer::new(1);
    let grid = empty_grid();
    let mut rng = Rng::with_seed(42);
    let pos = GridPos { x: 3, y: 3 };

    // Without bombs, should prefer valid moves
    let action = player.select_action(&grid, pos, &[], &mut rng);
    if action != BomberAction::Bomb && action != BomberAction::Wait {
        let target = move_target(&action, pos);
        assert!(grid.is_walkable(target.x, target.y));
    }
}

#[test]
fn test_validator_player_rejects_unsafe() {
    let mut player = ValidatorPlayer::new(2);
    let grid = empty_grid();
    let mut rng = Rng::with_seed(42);
    let pos = GridPos { x: 3, y: 3 };

    // With a bomb aimed at us, should avoid blast zone
    let events = vec![GameEvent::BombPlaced {
        player: 0,
        pos: (3, 1),
    }];
    player.known_bombs = vec![((3, 1), 2, BOMB_FUSE_TICKS)];

    let action = player.select_action(&grid, pos, &events, &mut rng);
    // Should not move into blast zone (3,1 has range 2, so (3,3) is in blast)
    // The player at (3,3) is in blast zone — should try to escape
    if action != BomberAction::Bomb && action != BomberAction::Wait {
        let target = move_target(&action, pos);
        // Moving out of blast zone is preferred
        assert!(
            target.x != 3 || target.y < 1 || target.y > 3,
            "Validator should escape blast zone, moved to ({},{})",
            target.x,
            target.y,
        );
    }
}

#[test]
fn test_hl_player_adapts() {
    let mut player = HLPlayer::new(3);
    let _grid = empty_grid();
    let _rng = Rng::with_seed(42);
    let _pos = GridPos { x: 3, y: 3 };

    // Simulate several rounds with good outcomes for Up
    for _ in 0..25 {
        player.round_actions.clear();
        // Push Up as the only action for this round
        player.round_actions.push(BomberAction::Up);
        player.update_outcome(true, false, 0);
    }

    // Q-value for Up should be positive
    let up_idx = action_index(&BomberAction::Up);
    assert!(
        player.q_values[up_idx] > 0.0,
        "HL should learn Up is good, Q={}",
        player.q_values[up_idx],
    );
}
