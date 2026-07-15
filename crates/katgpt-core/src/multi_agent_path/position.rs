//! Position trait — the spatial vocabulary seam (Plan 440 T1.8).
//!
//! `Position` is generic over the spatial domain. The default impl covers 2D
//! grid cells `(usize, usize)` — the paper's domain. Consumers with 3D worlds
//! or NavMesh topologies implement this trait; the rest of the substrate is
//! domain-agnostic.

use crate::sigmoid;

/// A position on the navigation graph.
///
/// Must be `Eq + Hash + Clone` so PIBT and the space-time A* can track visited
/// configurations in hash maps. The `neighbors()` method defines the graph
/// topology (4-connected grid by default; NavMesh adjacency for seal-core).
///
/// **Raw vs latent:** positions are always **raw** (physical, synced,
/// deterministic-replay-safe). The latent guidance field `Φ` is computed from
/// positions but is itself latent — see the sync-boundary rule in AGENTS.md.
pub trait Position: Eq + std::hash::Hash + Clone + std::fmt::Debug {
    /// Neighbors reachable in one step (including self for "wait").
    ///
    /// The returned slice must include the position itself if waiting in place
    /// is legal (it always is in LMAPF). Order is caller-defined; PIBT sorts
    /// candidates by its own cost, so neighbor order doesn't affect correctness.
    fn neighbors(&self) -> Vec<Self>;

    /// Whether this position is passable (not a wall/obstacle).
    fn is_passable(&self) -> bool {
        true
    }

    /// Heuristic distance to a goal position (for A* / PIBT tiebreak).
    ///
    /// Must be admissible (never overestimate true cost) for A* optimality.
    /// For grids this is Manhattan or Chebyshev distance. Default: 0 (becomes
    /// Dijkstra — correct but slower; callers should override).
    fn dist_heuristic(&self, goal: &Self) -> f32 {
        let _ = goal;
        0.0
    }
}

// ─────────────────────────────────────────────────────────────────────
// Default impl: 2D grid cell
// ─────────────────────────────────────────────────────────────────────

/// 2D grid cell — the paper's default domain.
///
/// 4-connected (up/down/left/right + wait). Manhattan distance heuristic.
/// Passability is checked against the grid's walls via [`GridMap`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GridPos {
    pub x: usize,
    pub y: usize,
}

impl GridPos {
    #[inline]
    pub fn new(x: usize, y: usize) -> Self {
        Self { x, y }
    }
}

impl Position for GridPos {
    fn neighbors(&self) -> Vec<Self> {
        // Wait + 4-connected moves. Negative-coord neighbors are omitted
        // (usize can't go below 0; the grid boundary handles edges).
        let mut v = Vec::with_capacity(5);
        v.push(*self); // wait
        if self.x > 0 {
            v.push(Self::new(self.x - 1, self.y));
        }
        v.push(Self::new(self.x + 1, self.y));
        if self.y > 0 {
            v.push(Self::new(self.x, self.y - 1));
        }
        v.push(Self::new(self.x, self.y + 1));
        v
    }

    fn dist_heuristic(&self, goal: &Self) -> f32 {
        let dx = (self.x as isize - goal.x as isize).unsigned_abs() as f32;
        let dy = (self.y as isize - goal.y as isize).unsigned_abs() as f32;
        dx + dy // Manhattan
    }
}

/// A grid map with explicit walls, for testing and benchmarks.
///
/// This is a test/bench helper — the substrate itself is generic over
/// `Position` and does not depend on this type. Consumers with their own
/// map representation (NavMesh, heightfield) implement `Position` directly.
#[derive(Clone)]
pub struct GridMap {
    pub width: usize,
    pub height: usize,
    /// `walls[y][x] == true` means blocked.
    pub walls: Vec<Vec<bool>>,
}

impl GridMap {
    /// Empty grid (no walls).
    pub fn empty(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            walls: vec![vec![false; width]; height],
        }
    }

    /// Mark a cell as a wall.
    pub fn set_wall(&mut self, x: usize, y: usize) {
        if x < self.width && y < self.height {
            self.walls[y][x] = true;
        }
    }

    /// Is the cell in-bounds and not a wall?
    #[inline]
    pub fn is_passable(&self, x: usize, y: usize) -> bool {
        x < self.width && y < self.height && !self.walls[y][x]
    }

    /// Clamp a [`GridPos`] to passable neighbors of `pos`.
    ///
    /// Returns only neighbors that are in-bounds and non-wall. Used by
    /// PIBT/guidance when the map has walls.
    pub fn passable_neighbors(&self, pos: &GridPos) -> Vec<GridPos> {
        let mut v = Vec::with_capacity(5);
        if self.is_passable(pos.x, pos.y) {
            v.push(*pos); // wait
        }
        if pos.x > 0 && self.is_passable(pos.x - 1, pos.y) {
            v.push(GridPos::new(pos.x - 1, pos.y));
        }
        if self.is_passable(pos.x + 1, pos.y) {
            v.push(GridPos::new(pos.x + 1, pos.y));
        }
        if pos.y > 0 && self.is_passable(pos.x, pos.y - 1) {
            v.push(GridPos::new(pos.x, pos.y - 1));
        }
        if self.is_passable(pos.x, pos.y + 1) {
            v.push(GridPos::new(pos.x, pos.y + 1));
        }
        v
    }
}

/// Sigmoid-gated passability soft-cost for bridge testing.
///
/// Given a raw scalar signal `s` (e.g. slope, threat density), returns a
/// soft penalty in `[0, 1)`. This is the canonical raw→latent bridge for
/// cost functions — a consumer's `CostFn` impl may call this to turn a raw
/// terrain signal into a latent cost contribution.
#[inline]
pub fn soft_cost(s: f32, beta: f32) -> f32 {
    sigmoid(s * beta)
}
