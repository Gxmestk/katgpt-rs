//! PUCT (Predictor + UCB applied to Trees) search — the AlphaZero recipe,
//! ported from `katgpt_pruners::go::moka_net::GoPuctMokaPlayer` (Plan 205) for
//! the standalone WASM build (Issue 204). Same algorithm, same vendored
//! weights, same forward pass — only the board type changed (this crate's
//! minimal `Board` instead of the full `GoState` engine).
//!
//! Exists to answer the measurement question the prior doc sidestepped:
//! does combining PUCT (98% strength, native) with WASM SIMD (0.5 ms/pass,
//! browser) produce a build that is BOTH strong AND fast, or does the
//! combination invert both headlines? See `.issues/204` and the
//! `go_arena.md` Table-A-vs-Table-B framing.
//!
//! Kept in sync by construction with the native PUCT implementation: this
//! is the same search loop, not a reimplementation. The native crate's
//! parity testing (Bench 205) covers the algorithm; this crate's job is
//! the latency measurement under WASM.

use crate::board::{AREA as BOARD_AREA, Board, SIZE as BOARD_SIZE, flood_group};
use crate::moka::{self, MokaScratch, MokaWeights};

/// A move: `Some(idx)` = play at board index `idx`; `None` = pass.
/// Replaces `GoAction` from the native crate.
type Move = Option<usize>;

struct PuctNode {
    /// Move that led to this node. `None` (= pass) for root by convention.
    action: Move,
    /// Board state at this node (cloned on expansion — the dominant non-SIMD
    /// cost in WASM, since `Vec<Cell>` heap-allocates per clone).
    state: Board,
    /// Visit count.
    visits: u32,
    /// Accumulated value from the perspective of the player who MOVED INTO
    /// this node (i.e., the parent's to_play). Negamax: negate at each level.
    total_value: f32,
    /// Policy prior P(s,a) from the parent's policy head evaluation.
    prior: f32,
    /// Arena indices of children.
    children: Vec<usize>,
    /// Arena index of parent. `None` for root.
    parent: Option<usize>,
    /// Whether this node has been expanded (policy+value evaluated, children
    /// created, or flagged terminal).
    expanded: bool,
}

impl PuctNode {
    fn new_root(state: Board) -> Self {
        Self {
            action: None,
            state,
            visits: 0,
            total_value: 0.0,
            prior: 1.0,
            children: Vec::new(),
            parent: None,
            expanded: false,
        }
    }

    #[inline]
    fn mean_value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.total_value / self.visits as f32
        }
    }
}

/// Standalone PUCT player. Mirrors `GoPuctMokaPlayer` field-for-field minus
/// the `GoPlayer` trait plumbing (this crate has no such trait).
pub struct PuctPlayer {
    weights: MokaWeights,
    scratch: MokaScratch,
    budget: usize,
    c_puct: f32,
    top_k: usize,
    /// Reused across `select_move` calls — avoids re-allocating the arena.
    arena: Vec<PuctNode>,
    /// Reused scratch for feature encoding (avoids per-expansion alloc).
    features_buf: Vec<f32>,
    nodes_evaluated: usize,
}

impl PuctPlayer {
    pub fn new(budget: usize, c_puct: f32, top_k: usize) -> Self {
        Self {
            weights: MokaWeights::load(),
            scratch: MokaScratch::new(),
            budget: budget.max(1),
            c_puct,
            top_k: top_k.max(1),
            arena: Vec::new(),
            features_buf: vec![0.0; moka::INPUT_ELEMENT_COUNT],
            nodes_evaluated: 0,
        }
    }

    #[inline]
    pub fn nodes_evaluated(&self) -> usize {
        self.nodes_evaluated
    }

    /// Expand a node: run policy+value, create children for top_k legal moves.
    /// Returns the value-head evaluation [-1,1] from this node's to_play
    /// perspective (or the exact terminal reward mapped onto the same range).
    fn expand(&mut self, node_idx: usize) -> f32 {
        // Collect parent chain BEFORE mutable borrow (Moka needs last-2-plies
        // history). Each ancestor's `action` is the move that produced it.
        let mut hist: Vec<Option<(usize, usize)>> = Vec::with_capacity(2);
        {
            let mut chain_actions: Vec<Move> = Vec::with_capacity(2);
            let mut cur = Some(node_idx);
            while let Some(idx) = cur {
                if chain_actions.len() >= 2 {
                    break;
                }
                let n = &self.arena[idx];
                if n.parent.is_some() {
                    chain_actions.push(n.action);
                }
                cur = n.parent;
            }
            for a in chain_actions.iter().rev() {
                hist.push((*a).map(|idx| (idx / BOARD_SIZE, idx % BOARD_SIZE)));
            }
        }

        let node = &mut self.arena[node_idx];
        node.expanded = true;

        if node.state.is_game_over() {
            // Map {0,1} reward onto [-1,+1] for the value-head's tanh range.
            return 2.0 * node.state.reward(node.state.to_play) - 1.0;
        }

        // Snapshot everything we need from `node`, then drop the mutable borrow
        // before re-entering the arena (children push needs `&mut self.arena`).
        let parent_state = node.state.clone();
        let legal: Vec<usize> = (0..BOARD_AREA).filter(|&i| parent_state.is_legal(i)).collect();

        // Encode features from the parent position + reconstructed history,
        // then run ONE forward pass. Reuses the persistent `features_buf` so
        // no allocation happens here (the dominant non-SIMD cost is the
        // `parent_state.clone()` above, not this).
        moka::encode_features_into(&parent_state, &hist, &mut self.features_buf);
        let (policy, value) =
            moka::forward_with_scratch(&self.weights, &self.features_buf, &mut self.scratch);
        self.nodes_evaluated += 1;

        // Rank moves by raw policy logit, keep top_k (always including pass as
        // a candidate — Moka's policy index BOARD_AREA is pass).
        let mut scored: Vec<(f32, Move)> = legal
            .iter()
            .map(|&i| (policy[i], Some(i)))
            .collect();
        scored.push((policy[BOARD_AREA], None));
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(self.top_k);

        // Softmax the top_k priors for normalized P(s,a).
        let max_logit = scored.iter().map(|(l, _)| *l).fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = scored.iter().map(|(l, _)| (l - max_logit).exp()).sum();
        let inv_exp_sum = if exp_sum > 0.0 { 1.0 / exp_sum } else { 1.0 };

        // Mutable borrow of `node` is now over (scored is owned). Push children.
        let children_start = self.arena.len();
        for (logit, action) in &scored {
            let prior = (logit - max_logit).exp() * inv_exp_sum;
            let mut child_state = parent_state.clone();
            match *action {
                Some(idx) => child_state.play(idx),
                None => child_state.pass(),
            }
            self.arena.push(PuctNode {
                action: *action,
                state: child_state,
                visits: 0,
                total_value: 0.0,
                prior,
                children: Vec::new(),
                parent: Some(node_idx),
                expanded: false,
            });
        }
        let children_end = self.arena.len();
        self.arena[node_idx].children.extend(children_start..children_end);

        value
    }

    /// Selection: traverse from root to first unexpanded leaf using PUCT.
    /// Returns the leaf node index.
    fn select(&self, root: usize) -> usize {
        let mut cur = root;
        loop {
            let node = &self.arena[cur];
            if !node.expanded || node.children.is_empty() {
                return cur;
            }
            // Pick child with highest PUCT score.
            // Q is negated because child.total_value is from child.to_play's
            // perspective, but we're selecting from parent's perspective.
            let parent_visits = node.visits.max(1) as f32;
            let sqrt_parent = parent_visits.sqrt();
            let mut best_idx = node.children[0];
            let mut best_score = f32::NEG_INFINITY;
            for &child_idx in &node.children {
                let child = &self.arena[child_idx];
                let q = -child.mean_value(); // negate: child's perspective → parent's
                let u = self.c_puct * child.prior * sqrt_parent / (1.0 + child.visits as f32);
                let score = q + u;
                if score > best_score {
                    best_score = score;
                    best_idx = child_idx;
                }
            }
            cur = best_idx;
        }
    }

    /// Backpropagate value from leaf to root. Negamax: negate at each level.
    fn backprop(&mut self, leaf_idx: usize, mut value: f32) {
        let mut cur = Some(leaf_idx);
        while let Some(idx) = cur {
            let node = &mut self.arena[idx];
            node.visits += 1;
            node.total_value += value;
            value = -value; // negate for parent (opponent's perspective)
            cur = node.parent;
        }
    }

    /// Run a full PUCT search from `state` and return the chosen move
    /// (`Some(idx)` or `None` for pass). Mirrors `GoPuctMokaPlayer::select_move`
    /// minus the trait plumbing. If `state` has no legal non-pass move AND pass
    /// would be pointless, returns `None` (pass).
    pub fn select_move(&mut self, state: &Board) -> Move {
        // Reset arena for this move's search.
        self.arena.clear();
        self.arena.push(PuctNode::new_root(state.clone()));
        let root = 0;

        for _ in 0..self.budget {
            // 1. Selection
            let leaf = self.select(root);
            // 2. Expansion + Evaluation
            let value = self.expand(leaf);
            // 3. Backpropagation (negamax)
            self.backprop(leaf, value);
        }

        // Pick most-visited child at root (AlphaZero convention).
        let root_node = &self.arena[root];
        let mut best_action: Move = None;
        let mut best_visits = 0u32;
        for &child_idx in &root_node.children {
            let child = &self.arena[child_idx];
            if child.visits > best_visits {
                best_visits = child.visits;
                best_action = child.action;
            }
        }

        best_action
    }
}

// `flood_group` is re-exported-by-use from `board`; reference it so the unused
// import warning stays quiet if future changes remove the only other user.
#[allow(dead_code)]
fn _ensure_flood_group_linked() {
    let _ = flood_group;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Cell;

    #[test]
    fn puct_returns_a_legal_move_from_empty_board() {
        let mut player = PuctPlayer::new(50, 1.5, 8);
        let board = Board::new();
        let mv = player.select_move(&board);
        // First move on an empty 9×9 must be a real placement (not pass) —
        // the policy head is trained on real openings, never pass on move 1.
        assert!(mv.is_some(), "PUCT on empty board should not pass");
        assert!(board.is_legal(mv.unwrap()));
        // At least one forward pass per budget iteration (terminal short-
        // circuits don't fire on an empty board).
        assert!(player.nodes_evaluated() > 0);
    }

    #[test]
    fn puct_search_is_deterministic_given_same_board() {
        // PUCT with no RNG in the loop (deterministic selection + argmax
        // final pick) must return the same move for the same input across
        // two runs. This is the WASM-side analog of the native G1 gate.
        let mut player = PuctPlayer::new(50, 1.5, 8);
        let mut board = Board::new();
        // Play a few moves to reach a non-trivial mid-game-ish position.
        board.play(40); // center-ish
        board.play(41);
        board.play(31);
        board.play(50);
        let first = player.select_move(&board);
        // Re-run from identical state — arena was cleared by the second call.
        let second = player.select_move(&board);
        assert_eq!(first, second, "PUCT must be deterministic given fixed input");
    }

    #[test]
    fn puct_budget_scales_nodes_evaluated() {
        // Sanity that the budget knob actually controls simulation count:
        // budget=100 should evaluate roughly 2× the nodes of budget=50
        // (not exact — terminals short-circuit — but same order of magnitude).
        let mut board = Board::new();
        board.play(40);
        board.play(41);

        let mut p50 = PuctPlayer::new(50, 1.5, 8);
        let _ = p50.select_move(&board);
        let n50 = p50.nodes_evaluated();

        let mut p100 = PuctPlayer::new(100, 1.5, 8);
        let _ = p100.select_move(&board);
        let n100 = p100.nodes_evaluated();

        assert!(n100 > n50, "budget=100 ({n100}) must exceed budget=50 ({n50})");
        // Allow generous slack for terminal short-circuits in late positions.
        assert!(n100 >= 2 * n50 - 30, "budget scaling off: 100→{n100}, 50→{n50}");
    }

    #[test]
    fn board_terminal_detection_after_double_pass() {
        let mut board = Board::new();
        assert!(!board.is_game_over());
        board.pass();
        assert!(!board.is_game_over()); // single pass
        board.pass();
        assert!(board.is_game_over()); // double pass ends game
        // A real move resets the consecutive-pass counter.
        let mut board2 = Board::new();
        board2.pass();
        board2.play(0);
        assert!(!board2.is_game_over());
    }

    #[test]
    fn board_reward_signs_are_consistent() {
        // On an empty board with komi 7.5 favoring White, both colors score
        // 0 stones and 0 territory, so White (7.5) is ahead → reward(White)=1,
        // reward(Black)=0.
        let board = Board::new();
        assert_eq!(board.reward(Cell::White), 1.0, "empty board: White wins on komi");
        assert_eq!(board.reward(Cell::Black), 0.0, "empty board: Black loses on komi");
    }
}
