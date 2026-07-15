//! Flow field — static, topology-aware direction assignment for Guided-PIBT
//! (Issue 149, Plan 440 T1.11).
//!
//! Distilled from the Guided-PIBT concept in Okumura et al. 2022/2025 and the
//! LLLG paper (arXiv:2605.16855 §4.2 "flow direction assignment"). On maps with
//! 1-wide corridors (maze topology), head-on deadlocks are the dominant
//! throughput limiter — two agents entering a corridor from opposite ends
//! cannot pass each other, and one must back up (the swap technique from
//! Issue 144 hurts throughput) or wait (LaCAM escalation from Issue 143 is
//! bounded and often can't resolve it).
//!
//! The paper's solution: **assign a canonical one-way direction to each
//! corridor segment** so that traffic flows in only one direction within a
//! corridor. Agents moving against the assigned direction incur a flow-mismatch
//! penalty in the PIBT cost tuple. On open maps (no corridors), the flow field
//! is empty — zero penalty — so open-map throughput is unaffected.
//!
//! # Mechanism
//!
//! 1. **Corridor detection:** a cell is a corridor cell if it has exactly 2
//!    passable neighbors and those neighbors are on opposite sides (horizontal
//!    pair or vertical pair). Dead-end cells (1 neighbor) and junction cells
//!    (3+ neighbors) are NOT corridors — they have no directionality to enforce.
//! 2. **Direction assignment:** each corridor cell gets a direction `(axis, sign)`.
//!    The axis is the corridor's orientation (Horizontal/Vertical). The sign is
//!    +1 (positive direction: right for horizontal, down for vertical). This
//!    makes all corridors one-way in the positive direction.
//! 3. **Penalty:** for agent at `from` moving to `to`, if `to` is a corridor
//!    cell with direction `(axis, sign)`, the move's direction along that axis
//!    is computed; if it's opposite to `sign`, `flow_mismatch = 1`, else `0`.
//!
//! # Modelless
//!
//! Entirely heuristic — corridor detection is a closed-form check on neighbor
//! count, and direction assignment is a deterministic rule. No training.
//!
//! # Pluggable seam
//!
//! The [`FlowField`] trait is the extension point. The default impl [`NoFlow`]
//! returns 0 everywhere (paper-faithful for non-corridor maps). The concrete
//! grid impl [`GridFlowField`] handles [`GridPos`]. A consumer with a NavMesh
//! or 3D topology implements the trait directly.
//!
//! # Why this is the safe promotion
//!
//! Issue 147 found that promoting the *dynamic* hindrance term (counter-flow
//! awareness between agents) ahead of `goal_dist` risks regressing open maps
//! (because agents would detour to avoid blocking siblings). The *static* flow
//! field is safe because it's only non-zero on corridor maps — on open maps,
//! there are no corridor cells, so `flow_mismatch` is always 0, and the cost
//! tuple degenerates to the paper-faithful ordering.

use super::position::{GridMap, GridPos};

/// A compass direction along a corridor axis.
///
/// Horizontal corridors run left-right (agents should move in +x or -x).
/// Vertical corridors run up-down (agents should move in +y or -y).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CorridorAxis {
    /// Corridor runs along the x-axis (left-right).
    Horizontal,
    /// Corridor runs along the y-axis (up-down).
    Vertical,
}

/// The direction assigned to a single corridor cell: axis + sign.
///
/// `sign` is +1 (positive direction along axis: right for Horizontal, down for
/// Vertical) or -1 (negative direction: left/up).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlowDirection {
    pub axis: CorridorAxis,
    /// +1 or -1. Encoded as i8 for compactness.
    pub sign: i8,
}

impl FlowDirection {
    /// Compute the flow mismatch for a move from `from` to `to`.
    ///
    /// Returns 0 if the move aligns with (or is orthogonal to) the assigned
    /// direction. Returns 1 if the move is directly against the assigned
    /// direction (e.g., corridor says +x but agent moves -x).
    ///
    /// A "wait" move (`from == to`) always returns 0 — waiting doesn't violate
    /// the flow.
    pub fn mismatch(&self, from: &GridPos, to: &GridPos) -> u8 {
        if from == to {
            return 0;
        }
        let dx = to.x as i32 - from.x as i32;
        let dy = to.y as i32 - from.y as i32;
        match self.axis {
            CorridorAxis::Horizontal => {
                if dx == 0 {
                    // Vertical move in a horizontal corridor — orthogonal, no penalty.
                    return 0;
                }
                let move_sign: i32 = if dx > 0 { 1 } else { -1 };
                if move_sign != self.sign as i32 {
                    1
                } else {
                    0
                }
            }
            CorridorAxis::Vertical => {
                if dy == 0 {
                    // Horizontal move in a vertical corridor — orthogonal, no penalty.
                    return 0;
                }
                let move_sign: i32 = if dy > 0 { 1 } else { -1 };
                if move_sign != self.sign as i32 {
                    1
                } else {
                    0
                }
            }
        }
    }
}

/// Flow field — pluggable seam for Guided-PIBT direction assignment.
///
/// Returns the flow mismatch (0 or 1) for a proposed move `from → to`. A return
/// of 0 means the move is flow-consistent (or no flow is assigned at `to`); a
/// return of 1 means the move goes against the assigned corridor direction.
///
/// The default impl [`NoFlow`] always returns 0 (paper-faithful for non-maze
/// maps). The concrete grid impl [`GridFlowField`] assigns directions to
/// 1-wide corridors.
///
/// # Determinism
///
/// The same `(from, to)` pair must always produce the same result for a given
/// flow field instance, to preserve deterministic replay.
pub trait FlowField<P> {
    /// Flow mismatch for the move `from → to`. Returns 0 (aligned) or 1 (against).
    fn mismatch(&self, from: &P, to: &P) -> u8;
}

/// No-op flow field — always returns 0. The default.
///
/// Paper-faithful for open/random/warehouse maps where there are no 1-wide
/// corridors to assign directions to.
pub struct NoFlow;

impl Default for NoFlow {
    fn default() -> Self {
        Self
    }
}

impl<P> FlowField<P> for NoFlow {
    #[inline]
    fn mismatch(&self, _from: &P, _to: &P) -> u8 {
        0
    }
}

/// Grid flow field — precomputed one-way direction assignment for 1-wide
/// corridors.
///
/// Built once from a [`GridMap`] via [`GridFlowField::from_map`]. Stores a
/// direction per corridor cell in a `Vec<Option<FlowDirection>>` indexed by
/// `y * width + x`. Non-corridor cells store `None`.
///
/// # Corridor definition
///
/// A cell `(x, y)` is a corridor cell if:
/// - It is passable.
/// - It has exactly 2 passable neighbors among the 4-connected cells.
/// - Those 2 neighbors are on opposite sides: `(x-1, y)` and `(x+1, y)`
///   (horizontal corridor), or `(x, y-1)` and `(x, y+1)` (vertical corridor).
///
/// Cells with 1 neighbor (dead-ends), 3+ neighbors (junctions), or 2
/// non-opposite neighbors (corners) are NOT corridors.
///
/// # Direction assignment
///
/// All corridors are assigned sign=+1 (positive direction: right for Horizontal,
/// down for Vertical). This makes corridors strictly one-way in the positive
/// coordinate direction. A more sophisticated assignment (alternating
/// directions per segment, flow-balanced assignment) is a future enhancement.
pub struct GridFlowField {
    width: usize,
    /// Per-cell flow direction. Indexed by `y * width + x`. `None` for non-corridor cells.
    directions: Vec<Option<FlowDirection>>,
}

impl GridFlowField {
    /// Build a flow field from a grid map. Detects corridors and assigns
    /// one-way directions (sign=+1 for all corridors).
    pub fn from_map(map: &GridMap) -> Self {
        let w = map.width;
        let h = map.height;
        let mut directions: Vec<Option<FlowDirection>> = vec![None; w * h];

        for y in 0..h {
            for x in 0..w {
                if !map.is_passable(x, y) {
                    continue;
                }
                let left = x > 0 && map.is_passable(x - 1, y);
                let right = x + 1 < w && map.is_passable(x + 1, y);
                let up = y > 0 && map.is_passable(x, y - 1);
                let down = y + 1 < h && map.is_passable(x, y + 1);

                let n_passable = (left as u8) + (right as u8) + (up as u8) + (down as u8);

                if n_passable != 2 {
                    continue;
                }

                let idx = y * w + x;
                if left && right && !up && !down {
                    // Horizontal corridor: left-right pair.
                    directions[idx] = Some(FlowDirection {
                        axis: CorridorAxis::Horizontal,
                        sign: 1,
                    });
                } else if up && down && !left && !right {
                    // Vertical corridor: up-down pair.
                    directions[idx] = Some(FlowDirection {
                        axis: CorridorAxis::Vertical,
                        sign: 1,
                    });
                }
                // else: corner (2 non-opposite neighbors) — not a corridor.
            }
        }

        Self {
            width: w,
            directions,
        }
    }

    /// Get the flow direction at a cell, if any.
    pub fn direction_at(&self, x: usize, y: usize) -> Option<FlowDirection> {
        if x >= self.width {
            return None;
        }
        let idx = y * self.width + x;
        self.directions.get(idx).copied().flatten()
    }

    /// Number of corridor cells detected.
    pub fn corridor_cell_count(&self) -> usize {
        self.directions.iter().filter(|d| d.is_some()).count()
    }
}

impl FlowField<GridPos> for GridFlowField {
    #[inline]
    fn mismatch(&self, from: &GridPos, to: &GridPos) -> u8 {
        // Check the destination cell's flow direction. If the destination is a
        // corridor cell, enforce its direction on the incoming move.
        let Some(dir) = self.direction_at(to.x, to.y) else {
            return 0;
        };
        dir.mismatch(from, to)
    }
}
