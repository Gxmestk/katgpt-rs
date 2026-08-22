//! Graph-rewriting primitives for the Cakiqi-Little syntactic identification
//! algorithm (Plan 457, Research 450).
//!
//! These are the "low-level" operations the recursive [`identify`]
//! composes: [`districts`] (c-components via bidirected edges),
//! [`ancestors_in_subgraph`] (directed-ancestor closure),
//! [`fix_node`] (the `Control_v ∘ Hide_v` operator), and
//! [`try_fixseq`] (greedy fixing sequence search).
//!
//! ## Why this is pure modelless
//!
//! Every function here is a pure closure over graph structure. The fix
//! operator in particular is the Cakiqi-Little distillation of the
//! Shpitser-Pearl ID algorithm's `do(X)` marginalisation: applying `fix(v)`
//! removes `v` and every edge touching it, equivalent to marginalising
//! `v`'s contribution out of the interventional distribution. No numerical
//! integration, no learned parameters.
//!
//! [`identify`]: super::identify::identify

use super::types::{Admg, NodeId};

// ────────────────────────────────────────────────────────────────────────────
// Adjacency queries
// ────────────────────────────────────────────────────────────────────────────

impl Admg {
    /// Directed parents of `v` in this graph.
    ///
    /// Allocates a fresh `Vec` per call. For the alloc-free inner loop of
    /// [`identify`](super::identify::identify), prefer
    /// [`Self::for_each_parent`].
    pub fn parents(&self, v: NodeId) -> Vec<NodeId> {
        let mut out = Vec::new();
        self.for_each_parent(v, |p| out.push(p));
        out
    }

    /// Push each directed parent of `v` into `out` (deduplicated). Allocation
    /// is the caller's responsibility.
    pub fn for_each_parent_into(&self, v: NodeId, out: &mut Vec<NodeId>) {
        for (p, c) in &self.directed {
            if *c == v && !out.contains(p) {
                out.push(*p);
            }
        }
    }

    /// Invoke `f(p)` for each directed parent of `v`. Alloc-free.
    pub fn for_each_parent<F: FnMut(NodeId)>(&self, v: NodeId, mut f: F) {
        for (p, c) in &self.directed {
            if *c == v {
                f(*p);
            }
        }
    }

    /// Bidirected neighbours of `v` (the other endpoint of each ↔ edge).
    pub fn bidir_neighbors(&self, v: NodeId) -> Vec<NodeId> {
        let mut out = Vec::new();
        for (a, b) in &self.bidirected {
            if *a == v {
                out.push(*b);
            } else if *b == v {
                out.push(*a);
            }
        }
        out
    }

    /// Invoke `f(other)` for each bidirected neighbour of `v`. Alloc-free.
    pub fn for_each_bidir_neighbor<F: FnMut(NodeId)>(&self, v: NodeId, mut f: F) {
        for (a, b) in &self.bidirected {
            if *a == v {
                f(*b);
            } else if *b == v {
                f(*a);
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Districts (c-components via bidirected edges)
// ────────────────────────────────────────────────────────────────────────────

impl Admg {
    /// The bidirected district of `v`: all nodes reachable from `v` via
    /// bidirected edges (including `v` itself). Uses a `Vec<NodeId>` as
    /// a visited-set — adequate for the bounded subgraph sizes the
    /// identification algorithm operates on (≤32 nodes per Plan 457 G2).
    pub fn district_of(&self, v: NodeId) -> Vec<NodeId> {
        let mut district: Vec<NodeId> = vec![v];
        let mut frontier: Vec<NodeId> = vec![v];
        // `next` is hoisted out of the BFS step loop and `clear`ed instead of
        // freshly allocated per frontier pop (one malloc/free per BFS step
        // before). Same push order, so `frontier` sees the same sequence.
        let mut next: Vec<NodeId> = Vec::new();
        while let Some(u) = frontier.pop() {
            next.clear();
            self.for_each_bidir_neighbor(u, |w| {
                if !district.contains(&w) {
                    district.push(w);
                    next.push(w);
                }
            });
            frontier.extend(next.iter().copied());
        }
        district
    }

    /// All bidirected districts (connected components via ↔ edges).
    /// Each node appears in exactly one district.
    pub fn districts(&self) -> Vec<Vec<NodeId>> {
        let mut visited: Vec<NodeId> = Vec::with_capacity(self.nodes.len());
        let mut out: Vec<Vec<NodeId>> = Vec::new();
        for &start in &self.nodes {
            if visited.contains(&start) {
                continue;
            }
            let d = self.district_of(start);
            visited.extend(d.iter().copied());
            out.push(d);
        }
        out
    }

    /// True iff `set` is a subset of the bidirected district of `v` in this
    /// graph. Used by [`try_fixseq`] to check fixability: a node is fixable
    /// iff its district is contained in the to-be-fixed set.
    pub fn district_of_contains_superset(&self, v: NodeId, superset: &[NodeId]) -> bool {
        let mut all_in = true;
        self.for_each_in_district(v, |w| {
            if !superset.contains(&w) {
                all_in = false;
            }
        });
        all_in
    }

    /// Invoke `f(w)` for every `w` in the bidirected district of `v`
    /// (including `v`). Alloc-free (modulo the `visited` Vec the caller
    /// passes, which is `clear`ed + reused).
    pub fn for_each_in_district_with_visited<F: FnMut(NodeId)>(
        &self,
        v: NodeId,
        visited: &mut Vec<NodeId>,
        mut f: F,
    ) {
        visited.clear();
        visited.push(v);
        let mut frontier: Vec<NodeId> = vec![v];
        f(v);
        while let Some(u) = frontier.pop() {
            self.for_each_bidir_neighbor(u, |w| {
                if !visited.contains(&w) {
                    visited.push(w);
                    frontier.push(w);
                    f(w);
                }
            });
        }
    }

    /// Convenience: invoke `f(w)` for every `w` in the bidirected district
    /// of `v` (including `v`). Allocates an internal visited-set.
    pub fn for_each_in_district<F: FnMut(NodeId)>(&self, v: NodeId, f: F) {
        let mut visited: Vec<NodeId> = Vec::new();
        self.for_each_in_district_with_visited(v, &mut visited, f);
    }

    /// Iterate over each bidirected district (connected component via ↔ edges).
    /// Alloc-free variant of [`Self::districts`]: the caller supplies 4 scratch
    /// `Vec`s (all `clear`ed + reused). `f` is invoked once per district with
    /// a slice view of the district's nodes.
    ///
    /// Contract: the 4 buffers MUST be distinct `Vec`s — aliasing them would
    /// corrupt the traversal. `district` and `next` in particular must not
    /// overlap because `district` accumulates the closure while `next` is the
    /// per-frontier-step work queue.
    ///
    /// Used by `identify_inner` step 4+5+6 (Plan 457 P4 zero-alloc refactor)
    /// to eliminate the dominant `districts()` allocator (~30 allocs/frame).
    pub fn for_each_district_with_buffers<F>(
        &self,
        visited: &mut Vec<NodeId>,
        district: &mut Vec<NodeId>,
        frontier: &mut Vec<NodeId>,
        next: &mut Vec<NodeId>,
        mut f: F,
    ) where
        F: FnMut(&[NodeId]),
    {
        visited.clear();
        for &start in &self.nodes {
            if visited.contains(&start) {
                continue;
            }
            // BFS the bidirected component containing `start` into `district`.
            district.clear();
            district.push(start);
            visited.push(start);
            frontier.clear();
            frontier.push(start);
            while let Some(u) = frontier.pop() {
                next.clear();
                self.for_each_bidir_neighbor(u, |w| {
                    if !visited.contains(&w) {
                        visited.push(w);
                        district.push(w);
                        next.push(w);
                    }
                });
                frontier.extend(next.iter().copied());
            }
            f(district);
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Directed ancestors
// ────────────────────────────────────────────────────────────────────────────

impl Admg {
    /// Directed ancestors of `seed` (including `seed` itself). Traverses
    /// parents only — bidirected edges are NOT causal ancestry.
    ///
    /// **Plan 457 Phase 1 T1.3**: the "ancestors in subgraph" primitive
    /// the recursive ID algorithm needs.
    pub fn ancestors(&self, seed: &[NodeId]) -> Vec<NodeId> {
        let mut an: Vec<NodeId> = seed.to_vec();
        let mut frontier: Vec<NodeId> = seed.to_vec();
        // `parents` is hoisted out of the BFS step loop and `clear`ed instead of
        // freshly allocated per frontier pop. Same push order into `frontier`.
        let mut parents: Vec<NodeId> = Vec::new();
        while let Some(v) = frontier.pop() {
            parents.clear();
            self.for_each_parent(v, |p| {
                if !an.contains(&p) {
                    an.push(p);
                    parents.push(p);
                }
            });
            frontier.extend(parents.iter().copied());
        }
        an
    }

    /// Same as [`Self::ancestors`] but writes into the caller-supplied
    /// `out` buffer (clear()ed first). Avoids per-call allocation.
    pub fn ancestors_into(&self, seed: &[NodeId], out: &mut Vec<NodeId>) {
        out.clear();
        out.extend(seed);
        let mut frontier: Vec<NodeId> = seed.to_vec();
        while let Some(v) = frontier.pop() {
            self.for_each_parent(v, |p| {
                if !out.contains(&p) {
                    out.push(p);
                    frontier.push(p);
                }
            });
        }
    }

    /// Fully alloc-free variant of [`Self::ancestors_into`] — the caller
    /// supplies both the output buffer AND a frontier (work-queue) buffer.
    /// Both buffers are `clear`ed + reused. Used by the inner loop of
    /// `identify_inner` (Issue 183 G4 refactor) to eliminate the frontier
    /// `Vec::with_capacity` that `ancestors_into` still pays.
    ///
    /// Contract: `out` and `frontier` MUST be distinct `Vec`s (aliasing
    /// them would corrupt the traversal).
    pub fn ancestors_with_frontier_into(
        &self,
        seed: &[NodeId],
        out: &mut Vec<NodeId>,
        frontier: &mut Vec<NodeId>,
    ) {
        out.clear();
        out.extend(seed);
        frontier.clear();
        frontier.extend(seed);
        while let Some(v) = frontier.pop() {
            self.for_each_parent(v, |p| {
                if !out.contains(&p) {
                    out.push(p);
                    frontier.push(p);
                }
            });
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Subgraph + Fix
// ────────────────────────────────────────────────────────────────────────────

impl Admg {
    /// Subgraph induced by `nodes` — keeps only nodes in the set and edges
    /// whose both endpoints are in the set.
    pub fn subgraph(&self, nodes: &[NodeId]) -> Admg {
        let mut g = Admg::new(
            self.nodes
                .iter()
                .copied()
                .filter(|n| nodes.contains(n))
                .collect(),
        );
        for &(p, c) in &self.directed {
            if nodes.contains(&p) && nodes.contains(&c) {
                g.directed.push((p, c));
            }
        }
        for &(a, b) in &self.bidirected {
            if nodes.contains(&a) && nodes.contains(&b) {
                g.bidirected.push((a, b));
            }
        }
        g
    }

    /// Alloc-free variant of [`Self::subgraph`] — writes into the
    /// caller-supplied `out` Admg (all three Vec fields `clear`ed + refilled).
    /// Used by the inner loop of `identify_inner` (Issue 183 G4 refactor) to
    /// eliminate the per-call `Admg::new` + grow allocations.
    pub fn subgraph_into(&self, nodes: &[NodeId], out: &mut Admg) {
        out.nodes.clear();
        out.nodes
            .extend(self.nodes.iter().copied().filter(|n| nodes.contains(n)));
        out.directed.clear();
        for &(p, c) in &self.directed {
            if nodes.contains(&p) && nodes.contains(&c) {
                out.directed.push((p, c));
            }
        }
        out.bidirected.clear();
        for &(a, b) in &self.bidirected {
            if nodes.contains(&a) && nodes.contains(&b) {
                out.bidirected.push((a, b));
            }
        }
    }

    /// Apply the `Fix_v` operation: remove `v` and every edge touching `v`
    /// (both directed and bidirected). This is the syntactic combined
    /// `Control_v ∘ Hide_v` per Cakiqi-Little §2.4.
    pub fn fix_node(&self, v: NodeId) -> Admg {
        let mut g = Admg::new(self.nodes.iter().copied().filter(|n| *n != v).collect());
        for &(p, c) in &self.directed {
            if p != v && c != v {
                g.directed.push((p, c));
            }
        }
        for &(a, b) in &self.bidirected {
            if a != v && b != v {
                g.bidirected.push((a, b));
            }
        }
        g
    }

    /// Alloc-free variant of [`Self::fix_node`] — writes into the caller-
    /// supplied `out` Admg (all three Vec fields `clear`ed + refilled).
    /// Used by `try_fixseq_into` (P4 zero-alloc refactor).
    pub fn fix_node_into(&self, v: NodeId, out: &mut Admg) {
        out.nodes.clear();
        out.nodes
            .extend(self.nodes.iter().copied().filter(|n| *n != v));
        out.directed.clear();
        for &(p, c) in &self.directed {
            if p != v && c != v {
                out.directed.push((p, c));
            }
        }
        out.bidirected.clear();
        for &(a, b) in &self.bidirected {
            if a != v && b != v {
                out.bidirected.push((a, b));
            }
        }
    }
}

/// Try to greedily fix every node in `w`. At each step, pick any node
/// `v ∈ W` (not yet fixed) whose bidirected district in the current graph
/// is entirely contained in `W`. Fix it (remove + drop edges). Repeat
/// until W is empty or no node is currently fixable.
///
/// **Soundness:** the order doesn't affect the existence of a valid
/// sequence. If no node is fixable at some step, no valid ordering exists.
/// (Cakiqi-Little §3.2.)
#[allow(clippy::result_large_err)] // NotIdentifiable is 129 bytes — only returned on rare error path.
pub fn try_fixseq(g: &Admg, w: &[NodeId]) -> Result<Admg, super::types::IdentificationError> {
    let mut current = g.clone();
    let mut remaining: Vec<NodeId> = w.to_vec();
    let mut progress = true;
    while progress && !remaining.is_empty() {
        progress = false;
        let mut next_v: Option<usize> = None;
        for (i, &v) in remaining.iter().enumerate() {
            if !current.contains_node(v) {
                next_v = Some(i);
                break;
            }
            // Check if district(v) ⊆ W.
            let dis_v = current.district_of(v);
            if dis_v.iter().all(|n| w.contains(n)) {
                next_v = Some(i);
                break;
            }
        }
        if let Some(i) = next_v {
            let v = remaining.remove(i);
            if current.contains_node(v) {
                current = current.fix_node(v);
            }
            progress = true;
        }
    }
    if remaining.is_empty() {
        Ok(current)
    } else {
        Err(super::types::IdentificationError::FixFailed {
            // Cause/effect echo unknown at this layer; identify() will wrap
            // with proper context if it propagates. Default to NodeId::from_u32(0)
            // is misleading, so we use a sentinel: from_label of "_fixseq".
            cause: NodeId::from_label(b"_fixseq"),
            effect: NodeId::from_label(b"_fixseq"),
        })
    }
}

/// Reusable workspace for [`try_fixseq_into`] — the zero-alloc variant of
/// [`try_fixseq`]. Holds a double-buffered current/next Admg (for
/// `fix_node_into` + `mem::swap`) plus the per-iteration district scratch
/// buffers.
///
/// Construct once and reuse across calls — all internal buffers `clear()`
/// on entry, so stale contents from a prior call are not observable.
#[derive(Default)]
pub struct FixSeqWorkspace {
    /// Current graph state. Initialized from `g` at entry.
    current: Admg,
    /// Double-buffer partner — receives the fix_node_into output, then swaps.
    next: Admg,
    /// Nodes in `W` not yet fixed.
    remaining: Vec<NodeId>,
    /// District-of-v scratch buffer (output).
    district: Vec<NodeId>,
    /// District BFS frontier (work queue).
    frontier: Vec<NodeId>,
    /// District BFS per-step next queue.
    next_buf: Vec<NodeId>,
}

impl FixSeqWorkspace {
    /// Fresh empty workspace. All Vec/Admg fields start at zero capacity —
    /// the first call grows them; subsequent calls reuse capacity.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Zero-alloc variant of [`try_fixseq`]: writes into the caller-supplied
/// `ws` workspace, returning `Ok(())` if the fix sequence exists. The
/// resulting fixed graph is NOT returned — callers that only need the
/// feasibility check (which is the case in `identify_inner` step 5) prefer
/// this variant because it avoids the 3-Vec return-path allocation.
///
/// Soundness is identical to [`try_fixseq`]: the order doesn't affect the
/// existence of a valid sequence (Cakiqi-Little §3.2).
///
/// Allocation budget: zero on the steady-state path. The first call grows
/// `ws.{current,next}.nodes/directed/bidirected` + 4 Vec scratch buffers;
/// subsequent calls reuse the capacity via `clear()`.
#[allow(clippy::result_large_err)] // NotIdentifiable is 129 bytes — only returned on rare error path.
pub fn try_fixseq_into(
    g: &Admg,
    w: &[NodeId],
    ws: &mut FixSeqWorkspace,
) -> Result<(), super::types::IdentificationError> {
    // Initialize current = g.clone() (reusing capacity).
    ws.current.nodes.clear();
    ws.current.nodes.extend(g.nodes.iter().copied());
    ws.current.directed.clear();
    ws.current.directed.extend(g.directed.iter().copied());
    ws.current.bidirected.clear();
    ws.current.bidirected.extend(g.bidirected.iter().copied());
    // next starts empty each call (cap reused).
    ws.next.nodes.clear();
    ws.next.directed.clear();
    ws.next.bidirected.clear();

    ws.remaining.clear();
    ws.remaining.extend(w.iter().copied());

    let mut progress = true;
    while progress && !ws.remaining.is_empty() {
        progress = false;
        let mut next_v: Option<usize> = None;
        // Split-borrow ws so the borrow checker can see the disjoint fields
        // (current is read by compute_district_into; district/frontier/
        // next_buf are written).
        let ws_ref = &mut *ws;
        for (i, &v) in ws_ref.remaining.iter().enumerate() {
            if !ws_ref.current.contains_node(v) {
                next_v = Some(i);
                break;
            }
            // Check district(v) ⊆ W, alloc-free using the workspace scratch.
            compute_district_into(
                &ws_ref.current,
                v,
                &mut ws_ref.district,
                &mut ws_ref.frontier,
                &mut ws_ref.next_buf,
            );
            if ws_ref.district.iter().all(|n| w.contains(n)) {
                next_v = Some(i);
                break;
            }
        }
        if let Some(i) = next_v {
            let v = ws.remaining.remove(i);
            if ws.current.contains_node(v) {
                // Apply fix_node_into(current, v) → next, then swap so current
                // becomes the fixed graph for the next iteration. Split-borrow
                // ws so the borrow checker sees disjoint fields.
                let ws_ref = &mut *ws;
                ws_ref.current.fix_node_into(v, &mut ws_ref.next);
                std::mem::swap(&mut ws_ref.current, &mut ws_ref.next);
                // After swap, `current` holds the fixed graph; `next` holds
                // the pre-fix state (will be overwritten next iteration).
                ws_ref.next.nodes.clear();
                ws_ref.next.directed.clear();
                ws_ref.next.bidirected.clear();
            }
            progress = true;
        }
    }

    if ws.remaining.is_empty() {
        Ok(())
    } else {
        Err(super::types::IdentificationError::FixFailed {
            cause: NodeId::from_label(b"_fixseq"),
            effect: NodeId::from_label(b"_fixseq"),
        })
    }
}

/// Compute district(v) — the bidirected component containing `v` — writing
/// into `out`. Reusable BFS scratch: `frontier` is the work queue, `next` is
/// the per-step expansion. All three buffers are `clear`ed + refilled.
fn compute_district_into(
    g: &Admg,
    v: NodeId,
    out: &mut Vec<NodeId>,
    frontier: &mut Vec<NodeId>,
    next: &mut Vec<NodeId>,
) {
    out.clear();
    out.push(v);
    frontier.clear();
    frontier.push(v);
    while let Some(u) = frontier.pop() {
        next.clear();
        g.for_each_bidir_neighbor(u, |w| {
            if !out.contains(&w) {
                out.push(w);
                next.push(w);
            }
        });
        frontier.extend(next.iter().copied());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_chain() -> Admg {
        // 0 → 1 → 2, plus 0 ↔ 2
        let n = |i: u32| NodeId::from_u32(i);
        let mut g = Admg::new(vec![n(0), n(1), n(2)]);
        g.directed_edge(n(0), n(1));
        g.directed_edge(n(1), n(2));
        g.bidirected_edge(n(0), n(2));
        g
    }

    #[test]
    fn parents_query_returns_directed_parents_only() {
        let g = build_chain();
        let n = |i: u32| NodeId::from_u32(i);
        let p1 = g.parents(n(1));
        assert_eq!(p1, vec![n(0)]);
        let p2 = g.parents(n(2));
        assert_eq!(p2, vec![n(1)]); // not n(0) — that's a bidirected neighbor
    }

    #[test]
    fn bidir_neighbors_query_returns_other_endpoints() {
        let g = build_chain();
        let n = |i: u32| NodeId::from_u32(i);
        let nbrs0 = g.bidir_neighbors(n(0));
        assert_eq!(nbrs0, vec![n(2)]);
        let nbrs2 = g.bidir_neighbors(n(2));
        assert_eq!(nbrs2, vec![n(0)]);
    }

    #[test]
    fn district_of_traverses_bidirected_edges() {
        let g = build_chain();
        let n = |i: u32| NodeId::from_u32(i);
        // 0 ↔ 2 are in the same district; 1 has no bidirected edges.
        let d0 = g.district_of(n(0));
        assert!(d0.contains(&n(0)) && d0.contains(&n(2)) && !d0.contains(&n(1)));
        let d1 = g.district_of(n(1));
        assert_eq!(d1, vec![n(1)]);
    }

    #[test]
    fn districts_partitions_nodes_by_bidirected_components() {
        let g = build_chain();
        let ds = g.districts();
        assert_eq!(ds.len(), 2, "two districts: {{0,2}} and {{1}}");
    }

    #[test]
    fn ancestors_traverses_directed_edges_only() {
        let g = build_chain();
        let n = |i: u32| NodeId::from_u32(i);
        let an = g.ancestors(&[n(2)]);
        // Ancestors of 2: 2 itself + parents 1 + parents of 1 = 0.
        // NOT including the bidirected neighbor 0 via the ↔ edge (but 0 is
        // a directed ancestor through 1 → 0... wait, 0 → 1 so 0 IS ancestor
        // of 2 through directed path).
        assert_eq!(an.len(), 3, "ancestors of 2 = {{0, 1, 2}}");
        assert!(an.contains(&n(0)));
        assert!(an.contains(&n(1)));
        assert!(an.contains(&n(2)));
    }

    #[test]
    fn subgraph_keeps_only_listed_nodes_and_internal_edges() {
        let g = build_chain();
        let n = |i: u32| NodeId::from_u32(i);
        let sub = g.subgraph(&[n(0), n(1)]);
        assert_eq!(sub.node_count(), 2);
        assert_eq!(sub.directed.len(), 1); // 0 → 1 kept; 1 → 2 dropped.
        assert_eq!(sub.bidirected.len(), 0); // 0 ↔ 2 dropped (2 not in set).
    }

    #[test]
    fn fix_node_removes_node_and_all_touching_edges() {
        let g = build_chain();
        let n = |i: u32| NodeId::from_u32(i);
        let fixed = g.fix_node(n(1));
        assert_eq!(fixed.node_count(), 2);
        assert_eq!(fixed.directed.len(), 0); // both 0→1 and 1→2 touch 1.
        assert_eq!(fixed.bidirected.len(), 1); // 0 ↔ 2 untouched.
    }

    #[test]
    fn try_fixseq_succeeds_when_all_nodes_are_fixable() {
        let g = build_chain();
        let n = |i: u32| NodeId::from_u32(i);
        // Fix node 1: its district is {1} (no bidirected edges) ⊆ {1}.
        let result = try_fixseq(&g, &[n(1)]);
        assert!(result.is_ok());
        let fixed = result.unwrap();
        assert_eq!(fixed.node_count(), 2); // 0 and 2 remain.
    }

    #[test]
    fn try_fixseq_fails_when_district_straddles_the_fix_set() {
        let g = build_chain();
        let n = |i: u32| NodeId::from_u32(i);
        // Try to fix node 0: its district is {0, 2}, not ⊆ {0}.
        // The algorithm should refuse (return Err).
        let result = try_fixseq(&g, &[n(0)]);
        assert!(result.is_err());
    }

    // ── P4 zero-alloc primitive drift tests (Plan 457) ───────────────────

    #[test]
    fn for_each_district_with_buffers_matches_districts() {
        // Drift test: the alloc-free callback API must enumerate the same
        // districts as the allocating `districts()` method, with the same
        // membership (order within a district is stable per node iteration
        // order; we compare as sorted sets).
        let g = build_chain();
        let n = |i: u32| NodeId::from_u32(i);

        let mut reference: Vec<std::collections::BTreeSet<NodeId>> = g
            .districts()
            .into_iter()
            .map(|d| d.into_iter().collect())
            .collect();
        reference.sort();

        let mut visited = Vec::new();
        let mut district = Vec::new();
        let mut frontier = Vec::new();
        let mut next = Vec::new();
        let mut got: Vec<std::collections::BTreeSet<NodeId>> = Vec::new();
        g.for_each_district_with_buffers(
            &mut visited,
            &mut district,
            &mut frontier,
            &mut next,
            |dist| {
                got.push(dist.iter().copied().collect());
            },
        );
        got.sort();

        assert_eq!(reference, got, "callback API must match districts()");

        // Re-run on the same buffers — must produce the same result
        // (verifies the clear-reuse contract).
        let mut got2: Vec<std::collections::BTreeSet<NodeId>> = Vec::new();
        g.for_each_district_with_buffers(
            &mut visited,
            &mut district,
            &mut frontier,
            &mut next,
            |dist| {
                got2.push(dist.iter().copied().collect());
            },
        );
        got2.sort();
        assert_eq!(reference, got2, "buffer reuse must not corrupt output");

        let _ = n; // silence unused closure warning on `n`
    }

    #[test]
    fn fix_node_into_matches_fix_node() {
        // Drift test: `fix_node_into` (alloc-free, caller-supplied out) must
        // produce the same graph as `fix_node` (allocating).
        let g = build_chain();
        let n = |i: u32| NodeId::from_u32(i);

        for v in [n(0), n(1), n(2)] {
            let reference = g.fix_node(v);
            let mut got = Admg::default();
            g.fix_node_into(v, &mut got);
            assert_eq!(
                reference.nodes,
                got.nodes,
                "nodes mismatch for v={:?}",
                v.as_u32()
            );
            assert_eq!(
                reference.directed,
                got.directed,
                "directed mismatch for v={:?}",
                v.as_u32()
            );
            assert_eq!(
                reference.bidirected,
                got.bidirected,
                "bidirected mismatch for v={:?}",
                v.as_u32()
            );
        }
    }

    #[test]
    fn try_fixseq_into_matches_try_fixseq() {
        // Drift test: `try_fixseq_into` (zero-alloc workspace variant) must
        // return the same Ok/Err verdict as `try_fixseq` (allocating variant).
        let g = build_chain();
        let n = |i: u32| NodeId::from_u32(i);

        let subsets_to_try: &[&[NodeId]] = &[
            &[n(0)],             // not fixable (district {0,2} ⊄ {0})
            &[n(1)],             // fixable (district {1})
            &[n(2)],             // not fixable (district {0,2} ⊄ {2})
            &[n(0), n(2)],       // fixable (district {0,2} ⊆ {0,2})
            &[n(1), n(2)],       // not fixable: 2's district {0,2} ⊄ {1,2}
            &[n(0), n(1), n(2)], // fixable (everything)
            &[],                 // trivially Ok (nothing to fix)
        ];

        let mut ws = FixSeqWorkspace::new();
        for w in subsets_to_try {
            let reference = try_fixseq(&g, w);
            let got = try_fixseq_into(&g, w, &mut ws);
            assert_eq!(
                reference.is_ok(),
                got.is_ok(),
                "verdict mismatch on w={:?}: reference={}, got={}",
                w,
                if reference.is_ok() { "Ok" } else { "Err" },
                if got.is_ok() { "Ok" } else { "Err" },
            );
        }

        // Re-run after the workspace has been used — buffers must still work.
        let got_again = try_fixseq_into(&g, &[n(1)], &mut ws);
        assert!(
            got_again.is_ok(),
            "workspace reuse must preserve correctness"
        );
    }

    #[test]
    fn for_each_district_with_buffers_handles_empty_graph() {
        let g = Admg::new(vec![]);
        let mut visited = Vec::new();
        let mut district = Vec::new();
        let mut frontier = Vec::new();
        let mut next = Vec::new();
        let mut count = 0;
        g.for_each_district_with_buffers(
            &mut visited,
            &mut district,
            &mut frontier,
            &mut next,
            |_| count += 1,
        );
        assert_eq!(count, 0, "empty graph has zero districts");
    }

    #[test]
    fn for_each_district_with_buffers_isolated_nodes() {
        // 3 nodes, no bidirected edges — 3 singleton districts.
        let n = |i: u32| NodeId::from_u32(i);
        let g = Admg::new(vec![n(0), n(1), n(2)]);
        let mut visited = Vec::new();
        let mut district = Vec::new();
        let mut frontier = Vec::new();
        let mut next = Vec::new();
        let mut count = 0;
        let mut all: Vec<NodeId> = Vec::new();
        g.for_each_district_with_buffers(
            &mut visited,
            &mut district,
            &mut frontier,
            &mut next,
            |dist| {
                count += 1;
                all.extend(dist.iter().copied());
            },
        );
        assert_eq!(count, 3);
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn fix_node_into_on_empty_graph() {
        let g = Admg::new(vec![]);
        let mut out = Admg::default();
        g.fix_node_into(NodeId::from_u32(0), &mut out);
        assert!(out.nodes.is_empty());
        assert!(out.directed.is_empty());
        assert!(out.bidirected.is_empty());
    }

    #[test]
    fn try_fixseq_into_empty_w_is_ok() {
        let g = build_chain();
        let mut ws = FixSeqWorkspace::new();
        let r = try_fixseq_into(&g, &[], &mut ws);
        assert!(r.is_ok(), "fixing empty set is trivially feasible");
    }
}
