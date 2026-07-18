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
        while let Some(u) = frontier.pop() {
            let mut next: Vec<NodeId> = Vec::new();
            self.for_each_bidir_neighbor(u, |w| {
                if !district.contains(&w) {
                    district.push(w);
                    next.push(w);
                }
            });
            frontier.extend(next);
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
    pub fn district_of_contains_superset(
        &self,
        v: NodeId,
        superset: &[NodeId],
    ) -> bool {
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
        while let Some(v) = frontier.pop() {
            let mut parents: Vec<NodeId> = Vec::new();
            self.for_each_parent(v, |p| {
                if !an.contains(&p) {
                    an.push(p);
                    parents.push(p);
                }
            });
            frontier.extend(parents);
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
        let mut g = Admg::new(self.nodes.iter().copied().filter(|n| nodes.contains(n)).collect());
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
        out.nodes.extend(self.nodes.iter().copied().filter(|n| nodes.contains(n)));
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
}
