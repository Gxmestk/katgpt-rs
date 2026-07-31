//! Subgraph extraction for the causal identification algorithm.
//!
//! ## Why this exists (Plan 457 key design decision #2)
//!
//! The Cakiqi-Little identification algorithm scales `O(k²)`–`O(k³)` in the
//! node count. A 1000-node game KG would take 100 ms–10 s — well outside
//! even an offline budget.
//!
//! The mitigation is subgraph extraction: identify over a 20-node relevant
//! subgraph (2-hop neighborhood of the query nodes) rather than the whole
//! KG. This is a heuristic — long confounder paths can be missed — but it
//! is correct for any confounder path within `hops` of the query.
//!
//! ## Default behaviour
//!
//! [`extract_relevant_subgraph`] defaults to `hops = 2`. This is the
//! "2-hop neighborhood" used by the Issue 545 PoC. The hop count is
//! configurable; callers needing deeper coverage can raise it at the cost
//! of larger subgraphs (and worse latency).

use super::types::{Admg, NodeId};

/// Extract the relevant subgraph for an identification query.
///
/// Performs a bounded BFS from `seeds`, expanding both directed and
/// bidirected edges (since either could carry a confounder path). Returns
/// the induced subgraph containing only nodes within `hops` of any seed.
///
/// `hops = 2` is the default used by Plan 457 — it captures the typical
/// "cause → mediator → effect" + "cause ↔ effect" structures without
/// exploding the subgraph on large KGs.
///
/// ## Example
///
/// ```
/// use katgpt_core::causal_id::{Admg, NodeId, extract_relevant_subgraph};
///
/// let n = |i: u32| NodeId::from_u32(i);
/// let mut g = Admg::new(vec![n(0), n(1), n(2), n(3), n(4)]);
/// g.directed_edge(n(0), n(1))
///  .directed_edge(n(1), n(2))
///  .directed_edge(n(2), n(3))
///  .directed_edge(n(3), n(4));
///
/// let sub = extract_relevant_subgraph(&g, &[n(0), n(4)], 2);
/// // 2 hops from {0, 4} covers {0, 1, 2} and {2, 3, 4} → all 5 nodes.
/// assert_eq!(sub.node_count(), 5);
/// ```
pub fn extract_relevant_subgraph(graph: &Admg, seeds: &[NodeId], hops: usize) -> Admg {
    if hops == 0 {
        return graph.subgraph(seeds);
    }
    let mut frontier: Vec<NodeId> = seeds.to_vec();
    let mut visited: Vec<NodeId> = seeds.to_vec();
    for _ in 0..hops {
        let mut next_frontier: Vec<NodeId> = Vec::new();
        for &v in &frontier {
            // Expand directed edges in both directions.
            graph.for_each_parent(v, |p| {
                if !visited.contains(&p) {
                    visited.push(p);
                    next_frontier.push(p);
                }
            });
            for (p, c) in &graph.directed {
                if *p == v && !visited.contains(c) {
                    visited.push(*c);
                    next_frontier.push(*c);
                }
            }
            // Expand bidirected edges.
            graph.for_each_bidir_neighbor(v, |w| {
                if !visited.contains(&w) {
                    visited.push(w);
                    next_frontier.push(w);
                }
            });
        }
        if next_frontier.is_empty() {
            break;
        }
        frontier = next_frontier;
    }
    graph.subgraph(&visited)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(i: u32) -> NodeId {
        NodeId::from_u32(i)
    }

    fn build_chain(n_nodes: u32) -> Admg {
        let nodes: Vec<NodeId> = (0..n_nodes).map(n).collect();
        let mut g = Admg::new(nodes);
        for i in 0..(n_nodes - 1) {
            g.directed_edge(n(i), n(i + 1));
        }
        g
    }

    #[test]
    fn hops_zero_returns_just_the_seeds() {
        let g = build_chain(5);
        let sub = extract_relevant_subgraph(&g, &[n(2)], 0);
        assert_eq!(sub.node_count(), 1);
    }

    #[test]
    fn hops_one_covers_immediate_neighbors_in_both_directions() {
        let g = build_chain(5);
        let sub = extract_relevant_subgraph(&g, &[n(2)], 1);
        // 1 hop from {2}: {1, 2, 3} (parent + child + self).
        assert_eq!(sub.node_count(), 3);
        assert!(sub.contains_node(n(1)));
        assert!(sub.contains_node(n(2)));
        assert!(sub.contains_node(n(3)));
    }

    #[test]
    fn hops_two_covers_2hop_neighborhood() {
        let g = build_chain(5);
        let sub = extract_relevant_subgraph(&g, &[n(2)], 2);
        // 2 hops from {2}: {0, 1, 2, 3, 4}.
        assert_eq!(sub.node_count(), 5);
    }

    #[test]
    fn multiple_seeds_union_their_neighborhoods() {
        let g = build_chain(10);
        let sub = extract_relevant_subgraph(&g, &[n(1), n(8)], 1);
        // 1 hop from {1} = {0, 1, 2}; 1 hop from {8} = {7, 8, 9}. Union = 6.
        assert_eq!(sub.node_count(), 6);
    }

    #[test]
    fn bidirected_edges_expand_the_frontier() {
        // 0 → 1 → 2 → 3 → 4 + 0 ↔ 4.
        let mut g = build_chain(5);
        g.bidirected_edge(n(0), n(4));
        // From seed {0}, hops=1 should pick up 4 via the bidirected edge.
        let sub = extract_relevant_subgraph(&g, &[n(0)], 1);
        assert!(
            sub.contains_node(n(4)),
            "bidirected edge should be traversed"
        );
    }

    #[test]
    fn out_of_graph_seeds_are_silently_ignored() {
        let g = build_chain(3);
        // Seed 99 is not in the graph; should be ignored without panicking.
        let sub = extract_relevant_subgraph(&g, &[n(0), n(99)], 1);
        assert!(sub.node_count() <= 3);
    }
}
