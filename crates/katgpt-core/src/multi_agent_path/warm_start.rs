//! Warm-start scheme — Π_{t-1}[2:w_Φ] suffix cache (Plan 440 T1.6).
//!
//! This is LLLG mechanism (b) from the paper (§3 "Leveraging LG in Receding
//! Horizon"). The guidance field `Φ_t` for the current tick can be
//! warm-started from the previous tick's state to reduce replanning cost and
//! improve temporal coherence.
//!
//! Three schemes (paper §4.C Fig. 6 ranking):
//!
//! | Scheme | Init `Φ_t` from | Paper result |
//! |---|---|---|
//! | [`WarmStartScheme::LllgPi`] | Suffix `Π_{t-1}[2:w_Φ]` of prev solution | **Best** — explicit collision-free forecast |
//! | [`WarmStartScheme::LllgPhi`] | Previous guidance `Φ_{t-1}` | Middle — soft bias only |
//! | [`WarmStartScheme::LllgEmpty`] | Empty (recompute from scratch) | Worst baseline |
//!
//! # Pluggable seam
//!
//! The enum is the extension point for a personality-weighted blend
//! (riir-ai/318 Extension C): curious NPCs use `LllgEmpty` (explore), while
//! conservative NPCs use `LllgPi` (stick to the plan). The default is
//! `LllgPi` per paper.

use super::position::Position;

/// Warm-start strategy for the guidance field.
///
/// Pluggable seam #3.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WarmStartScheme {
    /// Initialize `Φ_t` from the suffix `Π_{t-1}[2:w_Φ]` of the previous
    /// tick's solution. Paper default and best-performing scheme.
    ///
    /// The previous solution `Π_{t-1}` is a `w_Π`-step plan; we take positions
    /// `[2, w_Φ]` (skipping the already-executed first step) and pad with the
    /// last position if `w_Φ > w_Π - 1`.
    #[default]
    LllgPi,
    /// Initialize `Φ_t` with the previous tick's guidance `Φ_{t-1}`.
    /// A soft collision-avoidance bias — strictly weaker than `LllgPi`.
    LllgPhi,
    /// No warm-start — recompute `Φ_t` from scratch each tick.
    /// The baseline; useful for ablation.
    LllgEmpty,
}

/// The warm-start cache: stores the previous tick's solution `Π_{t-1}` and
/// guidance `Φ_{t-1}` for the next tick's initialization.
///
/// Owned by the [`LifelongLaCam`](super::LifelongLaCam) orchestrator. Updated
/// at the end of each tick.
pub struct WarmStartCache<P: Position> {
    scheme: WarmStartScheme,
    /// Previous tick's solution suffix (per-agent paths). `Π_{t-1}`.
    prev_solution: Vec<Vec<P>>,
    /// Previous tick's guidance field. `Φ_{t-1}`.
    prev_guidance: Vec<Vec<P>>,
    /// Window length `w_Φ` (for suffix extraction + padding).
    w_phi: usize,
}

impl<P: Position> WarmStartCache<P> {
    pub fn new(scheme: WarmStartScheme, w_phi: usize) -> Self {
        Self {
            scheme,
            prev_solution: Vec::new(),
            prev_guidance: Vec::new(),
            w_phi,
        }
    }

    /// Set the scheme (can be changed at runtime).
    pub fn set_scheme(&mut self, scheme: WarmStartScheme) {
        self.scheme = scheme;
    }

    /// Produce the warm-start initialization for the guidance field.
    ///
    /// Returns a per-agent initial path that the [`LocalGuidanceSource`] can
    /// optionally seed its occupancy map with. For [`WarmStartScheme::LllgEmpty`]
    /// this is empty (no seeding).
    pub fn warm_start(&self) -> Vec<Vec<P>> {
        match self.scheme {
            WarmStartScheme::LllgPi => self.prev_solution_suffix(),
            WarmStartScheme::LllgPhi => self.prev_guidance.clone(),
            WarmStartScheme::LllgEmpty => Vec::new(),
        }
    }

    /// Extract `Π_{t-1}[2:w_Φ]` with padding for `w_Φ > len(suffix)`.
    ///
    /// Skip index 0 (already executed), take up to `w_Φ` positions, pad with
    /// the last available position if the suffix is shorter than `w_Φ`.
    fn prev_solution_suffix(&self) -> Vec<Vec<P>> {
        self.prev_solution
            .iter()
            .map(|path| {
                let start = 1.min(path.len()); // skip executed step
                let suffix = &path[start..];
                let mut out = Vec::with_capacity(self.w_phi);
                // Fill with suffix positions.
                for p in suffix.iter().take(self.w_phi) {
                    out.push(p.clone());
                }
                // Pad with last position if too short.
                while out.len() < self.w_phi {
                    if let Some(last) = out.last() {
                        out.push(last.clone());
                    } else if let Some(first) = path.first() {
                        out.push(first.clone());
                    } else {
                        break;
                    }
                }
                out
            })
            .collect()
    }

    /// Record this tick's solution + guidance for the next tick's warm-start.
    ///
    /// Called by the orchestrator after PIBT produces the joint action and the
    /// guidance source produces `Φ_t`.
    pub fn record(&mut self, solution: Vec<Vec<P>>, guidance: Vec<Vec<P>>) {
        self.prev_solution = solution;
        self.prev_guidance = guidance;
    }

    /// Clear the cache (e.g. on cold-start or zone reset).
    pub fn clear(&mut self) {
        self.prev_solution.clear();
        self.prev_guidance.clear();
    }
}
