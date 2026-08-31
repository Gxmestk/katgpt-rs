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

    /// Flat index for O(1) occupancy lookups (Issue 516 T1a).
    ///
    /// Returns `Some(index)` if this position can be mapped to a flat array
    /// index given the grid `width`. For 2D grids this is `y * width + x`.
    /// Non-grid positions (NavMesh, 3D) return `None` — the HashMap path is
    /// used instead.
    ///
    /// The index MUST be stable for a given position + width (deterministic).
    /// Consumers configure flat occupancy via
    /// [`SpaceTimeGuidance::with_flat_occupancy`] or
    /// [`LocalGuidanceSource::ensure_flat_occupancy`].
    fn flat_index(&self, _width: usize) -> Option<usize> {
        None
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

    #[inline]
    fn flat_index(&self, width: usize) -> Option<usize> {
        Some(self.y * width + self.x)
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

    /// Parse a MovingAI benchmark map file (the standard 2D pathfinding
    /// format from https://movingai.com/benchmarks/grids.html).
    ///
    /// Format:
    /// ```text
    /// type octile
    /// height H
    /// width W
    /// map
    /// <H rows of W characters each>
    /// ```
    ///
    /// Per the MovingAI MAPF benchmark convention, only `.` is passable.
    /// All other characters (`@`, `O`, `T`, `W`, `S`, `V`, `U`, ...) are
    /// treated as walls. This matches how the LaCAM / LLLG papers configure
    /// these maps (4-connected, ground-traversable cells only).
    ///
    /// This is a reusable leaf primitive — any consumer or test loading a
    /// MovingAI map needs it, and the substrate itself is grid-format-
    /// agnostic (it works over `Position`). Returns `None` on malformed input
    /// so callers can fall back to a synthetic generator if a download fails.
    ///
    /// # Plan 440 Issue 148
    /// Added to load the real `ht_chantry.map` (162×141, Dragon Age: Origins)
    /// for a fair G1 comparison against the paper, replacing the synthetic
    /// `ht_chantry_approx` whose tight maze corridors capped throughput at
    /// ~1.5 (2× the real map's corridor density).
    pub fn from_movingai(text: &str) -> Option<Self> {
        let mut lines = text.lines();

        // Header: 4 lines. Be lenient about the `type` value (octile is by far
        // the most common, but the parser doesn't depend on it).
        let _type_line = lines.next()?;
        let height_line = lines.next()?;
        let width_line = lines.next()?;
        let map_marker = lines.next()?;

        if !height_line.starts_with("height") {
            return None;
        }
        if !width_line.starts_with("width") {
            return None;
        }
        if map_marker.trim() != "map" {
            return None;
        }

        let h: usize = height_line.split_whitespace().nth(1)?.parse().ok()?;
        let w: usize = width_line.split_whitespace().nth(1)?.parse().ok()?;

        if h == 0 || w == 0 {
            return None;
        }

        let mut map = Self::empty(w, h);
        for (y, row) in lines.take(h).enumerate() {
            for (x, c) in row.chars().take(w).enumerate() {
                // Only `.` is passable in the MAPF benchmark convention.
                if c != '.' {
                    map.set_wall(x, y);
                }
            }
        }
        Some(map)
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
