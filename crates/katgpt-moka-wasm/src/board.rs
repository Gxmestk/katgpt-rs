//! Minimal 9×9 Go board — just enough for legal self-play moves to generate
//! realistic benchmark positions. Not a general engine (no 13×13/19×19, no
//! scoring, no replay) — this crate exists purely to give the WASM benchmark
//! (Plan 565) a dependency-free way to drive `encode_features`, mirroring
//! what the real Moka JS harness does with its own `game.ts`.

pub const SIZE: usize = 9;
pub const AREA: usize = SIZE * SIZE;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cell {
    Empty,
    Black,
    White,
}

impl Cell {
    #[inline]
    pub fn opponent(self) -> Self {
        match self {
            Cell::Black => Cell::White,
            Cell::White => Cell::Black,
            Cell::Empty => panic!("opponent() on Empty"),
        }
    }
}

#[derive(Clone)]
pub struct Board {
    pub cells: Vec<Cell>,
    pub to_play: Cell,
    pub ko_point: Option<usize>,
}

fn neighbors(idx: usize) -> Vec<usize> {
    let row = idx / SIZE;
    let col = idx % SIZE;
    let mut out = Vec::with_capacity(4);
    if row > 0 {
        out.push(idx - SIZE);
    }
    if row + 1 < SIZE {
        out.push(idx + SIZE);
    }
    if col > 0 {
        out.push(idx - 1);
    }
    if col + 1 < SIZE {
        out.push(idx + 1);
    }
    out
}

/// Flood-fill the group containing `start`, returning (stones, liberty positions).
/// Mirrors `katgpt_pruners::go::utils::flood_group`.
pub fn flood_group(cells: &[Cell], start: usize) -> (Vec<usize>, Vec<usize>) {
    let color = cells[start];
    if color == Cell::Empty {
        return (Vec::new(), Vec::new());
    }
    let mut group = Vec::new();
    let mut liberties = Vec::new();
    let mut visited = [false; AREA];
    let mut stack = vec![start];
    while let Some(pos) = stack.pop() {
        if visited[pos] {
            continue;
        }
        visited[pos] = true;
        match cells[pos] {
            c if c == color => {
                group.push(pos);
                for n in neighbors(pos) {
                    if !visited[n] {
                        stack.push(n);
                    }
                }
            }
            Cell::Empty => liberties.push(pos),
            _ => {}
        }
    }
    (group, liberties)
}

impl Board {
    pub fn new() -> Self {
        Self {
            cells: vec![Cell::Empty; AREA],
            to_play: Cell::Black,
            ko_point: None,
        }
    }

    fn would_be_suicide(&self, idx: usize, color: Cell) -> bool {
        let mut trial = self.cells.clone();
        trial[idx] = color;
        // Does placing capture any opponent group?
        for n in neighbors(idx) {
            if trial[n] == color.opponent() {
                let (_, libs) = flood_group(&trial, n);
                if libs.is_empty() {
                    return false; // captures something — legal
                }
            }
        }
        // No capture — legal only if the placed stone's own group has a liberty.
        let (_, libs) = flood_group(&trial, idx);
        libs.is_empty()
    }

    pub fn is_legal(&self, idx: usize) -> bool {
        self.cells[idx] == Cell::Empty
            && self.ko_point != Some(idx)
            && !self.would_be_suicide(idx, self.to_play)
    }

    pub fn legal_moves(&self) -> Vec<usize> {
        (0..AREA).filter(|&i| self.is_legal(i)).collect()
    }

    /// Places a stone, resolves captures, updates ko/turn. Caller must have
    /// already checked `is_legal`.
    pub fn play(&mut self, idx: usize) {
        let color = self.to_play;
        let opponent = color.opponent();
        self.cells[idx] = color;

        let mut captured = Vec::new();
        let mut visited_opp = [false; AREA];
        for n in neighbors(idx) {
            if self.cells[n] == opponent && !visited_opp[n] {
                let (group, libs) = flood_group(&self.cells, n);
                for &g in &group {
                    visited_opp[g] = true;
                }
                if libs.is_empty() {
                    captured.extend(group);
                }
            }
        }
        for &c in &captured {
            self.cells[c] = Cell::Empty;
        }

        // Simple ko: exactly one stone captured, and the capturing stone's
        // own group is a single stone with a single liberty (the ko point).
        if captured.len() == 1 {
            let (my_group, my_libs) = flood_group(&self.cells, idx);
            self.ko_point = if my_group.len() == 1 && my_libs.len() == 1 {
                Some(captured[0])
            } else {
                None
            };
        } else {
            self.ko_point = None;
        }

        self.to_play = opponent;
    }

    pub fn pass(&mut self) {
        self.ko_point = None;
        self.to_play = self.to_play.opponent();
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}
