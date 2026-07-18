//! The top-level [`identify`] driver + the recursive Shpitser-Pearl ID
//! algorithm (Cakiqi-Little Theorem 1 distillation).
//!
//! ## The algorithm
//!
//! Given an ADMG `G`, an intervention set `A` (cause) and an effect set `Y`,
//! `identify(Y, do(A))` returns the interventional signature backbone
//! `Y⋆ = An(Y in G[V\A])` — the set of nodes the derivation had to consider.
//! Returns `Err(NotIdentifiable)` if there is no syntactic derivation
//! (the canonical hedge case).
//!
//! This is the **recursive Shpitser-Pearl ID algorithm** (2006) as
//! distilled into pure graph rewriting by Cakiqi-Little §3. The
//! Issue 545 PoC caught a soundness bug in the simpler one-pass formulation
//! (which computed districts of `G[Y⋆]` instead of `G[V]`); this module
//! implements the corrected recursive version.
//!
//! ## Six steps
//!
//! 1. `A = ∅` → trivially identifiable as the marginal.
//! 2. `V ≠ An(Y)` → restrict graph to `An(Y)`, recurse.
//! 3. `(V\A) \ An(Y in G[V\A]) ≠ ∅` → drop those nodes, recurse.
//! 4. `D = An(Y in G[V\A])`. Find districts of `G[V]` intersecting D.
//! 5. If exactly one district `C` contains all of D:
//!    - If `C = V` → FAIL (hedge / bow-arc).
//!    - Else fix `V \ C`. If unfixable → FAIL. Else return signature of D.
//! 6. Else (multiple districts intersect D): recurse on each, return union.
//!
//! ## What ships
//!
//! The full modelless primitive — `identify` returns the derivation
//! backbone. The interpretation (probabilistic vs deterministic vs
//! min-plus) is consumer-side and not implemented here.

use super::fixing::try_fixseq;
use super::types::{Admg, AdmgSignature, IdentificationError, NodeId};

/// Recursively identify `Σ_{Y|do(A)}` on graph `g`.
///
/// `cause` is `A`, `effect` is `Y`. Returns the interventional signature
/// backbone `Y⋆` on success, or [`IdentificationError::NotIdentifiable`]
/// if the query has no syntactic derivation (the canonical hedge case).
///
/// ## Example
///
/// ```
/// use katgpt_core::causal_id::{Admg, NodeId, identify};
///
/// let a = NodeId::from_u32(0);
/// let y = NodeId::from_u32(1);
/// let mut g = Admg::new(vec![a, y]);
/// g.directed_edge(a, y).bidirected_edge(a, y); // bow-arc
///
/// // Bow-arc is the canonical NOT-IDENTIFIABLE hedge.
/// let result = identify(&g, &[a], &[y]);
/// assert!(result.is_err());
/// ```
#[allow(clippy::result_large_err)] // NotIdentifiable is 129 bytes — only returned on rare error path.
pub fn identify(
    g: &Admg,
    cause: &[NodeId],
    effect: &[NodeId],
) -> Result<AdmgSignature, IdentificationError> {
    if cause.is_empty() || effect.is_empty() {
        return Err(IdentificationError::EmptyQuery);
    }
    let cause_head = cause[0];
    let effect_head = effect[0];
    identify_inner(g, cause, effect, 0, cause_head, effect_head)
}

/// Maximum recursion depth — defensive guard. The algorithm strictly
/// shrinks `V` or `A` at every step so depth is bounded by `|V|`, but the
/// guard catches any hypothetical bug.
const MAX_DEPTH: u32 = 64;

#[allow(clippy::result_large_err)] // NotIdentifiable is 129 bytes — only returned on rare error path.
fn identify_inner(
    g: &Admg,
    cause: &[NodeId],
    effect: &[NodeId],
    depth: u32,
    cause_head: NodeId,
    effect_head: NodeId,
) -> Result<AdmgSignature, IdentificationError> {
    if depth > MAX_DEPTH {
        return Err(IdentificationError::NotIdentifiable {
            cause: cause_head,
            effect: effect_head,
            hedge: None,
        });
    }

    let v: Vec<NodeId> = g.nodes.clone();

    // Step 1: empty intervention — trivially identifiable as the marginal.
    if cause.is_empty() {
        return Ok(AdmgSignature::from_nodes(effect.iter().copied()));
    }

    // Step 2: ancestry reduction. If V is not all ancestors of Y, restrict.
    let an_y = g.ancestors(effect);
    if an_y.len() != v.len() {
        let sub = g.subgraph(&an_y);
        let new_cause: Vec<NodeId> = cause.iter().copied().filter(|c| an_y.contains(c)).collect();
        return identify_inner(&sub, &new_cause, effect, depth + 1, cause_head, effect_head);
    }

    // Step 3: drop nodes in (V\A) that are not ancestors of Y in G[V\A].
    let v_minus_a: Vec<NodeId> = v.iter().copied().filter(|n| !cause.contains(n)).collect();
    let g_va = g.subgraph(&v_minus_a);
    let an_y_in_gva = g_va.ancestors(effect);
    let w: Vec<NodeId> = v_minus_a.iter().copied().filter(|n| !an_y_in_gva.contains(n)).collect();
    if !w.is_empty() {
        let new_v: Vec<NodeId> = v.iter().copied().filter(|n| !w.contains(n)).collect();
        let sub = g.subgraph(&new_v);
        return identify_inner(&sub, cause, effect, depth + 1, cause_head, effect_head);
    }

    // Step 4: D = An(Y in G[V\A]) = an_y_in_gva. Districts of G[V] intersecting D.
    let d: &[NodeId] = &an_y_in_gva;
    let all_districts = g.districts();
    let intersecting: Vec<&Vec<NodeId>> = all_districts
        .iter()
        .filter(|dist| dist.iter().any(|n| d.contains(n)))
        .collect();

    if intersecting.is_empty() {
        // Defensive: should not happen if effect ⊆ v.
        return Ok(AdmgSignature::from_nodes(effect.iter().copied()));
    }

    // Step 5: if exactly one district of G[V] contains all of D.
    if intersecting.len() == 1 {
        let c = intersecting[0];
        if d.iter().all(|n| c.contains(n)) {
            // FAIL condition: the c-component containing D is the entire V.
            // This is the bow-arc / hedge: cannot intervene outside C.
            if c.len() == v.len() && c.iter().all(|n| v.contains(n)) {
                return Err(IdentificationError::NotIdentifiable {
                    cause: cause_head,
                    effect: effect_head,
                    hedge: first_two_nodes(c),
                });
            }
            // Fix V \ C in G. This is the "back-door" branch.
            let fix_set: Vec<NodeId> = v.iter().copied().filter(|n| !c.contains(n)).collect();
            match try_fixseq(g, &fix_set) {
                Ok(_) => return Ok(AdmgSignature::from_nodes(d.iter().copied())),
                Err(_) => {
                    return Err(IdentificationError::NotIdentifiable {
                        cause: cause_head,
                        effect: effect_head,
                        hedge: first_two_nodes(c),
                    });
                }
            }
        }
        // else: fall through to step 6 (D spans multiple districts but only
        // one is intersecting — defensive, shouldn't happen for valid input).
    }

    // Step 6: multiple districts. Recurse on each to propagate any Err,
    // then return D as the signature backbone. The union of sub-problem
    // backbones equals D by the ID algorithm's correctness — we return D
    // directly to avoid accumulation drift across recursive splits.
    for c in &intersecting {
        let c_in_d: Vec<NodeId> = c.iter().copied().filter(|n| d.contains(n)).collect();
        if c_in_d.is_empty() {
            continue;
        }
        // New intervention set = V \ c_in_d (the original V minus this district).
        let new_cause: Vec<NodeId> = v.iter().copied().filter(|n| !c_in_d.contains(n)).collect();
        let _ = identify_inner(g, &new_cause, &c_in_d, depth + 1, cause_head, effect_head)?;
    }
    Ok(AdmgSignature::from_nodes(d.iter().copied()))
}

/// Pick the first two nodes from `set` for hedge diagnostics. Returns
/// `None` if the set has fewer than 2 nodes.
fn first_two_nodes(set: &[NodeId]) -> Option<(NodeId, NodeId)> {
    let mut iter = set.iter().copied();
    let a = iter.next()?;
    let b = iter.next()?;
    Some((a, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(i: u32) -> NodeId {
        NodeId::from_u32(i)
    }

    /// Scenario A — classic front-door. `A→M→Y, A↔Y`. Identifiable via
    /// front-door adjustment.
    fn scenario_a_frontdoor() -> (Admg, NodeId, NodeId) {
        let (a, m, y) = (n(0), n(1), n(2));
        let mut g = Admg::new(vec![a, m, y]);
        g.directed_edge(a, m).directed_edge(m, y).bidirected_edge(a, y);
        (g, a, y)
    }

    /// Scenario B — classic back-door. `Z→A→Y, Z→Y`. Identifiable via
    /// back-door adjustment on Z.
    fn scenario_b_backdoor() -> (Admg, NodeId, NodeId) {
        let (z, a, y) = (n(0), n(1), n(2));
        let mut g = Admg::new(vec![z, a, y]);
        g.directed_edge(z, a).directed_edge(a, y).directed_edge(z, y);
        (g, a, y)
    }

    /// Scenario C — game-world KG (13 nodes) with `NPC1 ↔ NPC2` confounder.
    /// The realistic load-bearing test.
    fn scenario_c_game_kg() -> (Admg, NodeId, NodeId) {
        let (f1, f2, f3) = (n(0), n(1), n(2));
        let (r1, r2) = (n(3), n(4));
        let (npc1, npc2, npc3) = (n(5), n(6), n(7));
        let (e1, e2, outcome) = (n(8), n(9), n(10));
        let (mood1, mood2) = (n(11), n(12));
        let mut g = Admg::new(vec![f1, f2, f3, r1, r2, npc1, npc2, npc3, e1, e2, outcome, mood1, mood2]);
        g.directed_edge(f1, npc1)
            .directed_edge(f2, npc2)
            .directed_edge(f3, npc3)
            .directed_edge(r1, npc1)
            .directed_edge(r2, npc2)
            .directed_edge(npc1, e1)
            .directed_edge(npc2, e2)
            .directed_edge(e1, outcome)
            .directed_edge(e2, outcome)
            .directed_edge(f1, mood1)
            .directed_edge(mood2, npc3);
        g.bidirected_edge(npc1, npc2);
        (g, e1, outcome)
    }

    /// Scenario D — bow-arc negative control. `A → Y, A ↔ Y`. NOT
    /// IDENTIFIABLE — the canonical hedge.
    fn scenario_d_bowarc() -> (Admg, NodeId, NodeId) {
        let (a, y) = (n(0), n(1));
        let mut g = Admg::new(vec![a, y]);
        g.directed_edge(a, y).bidirected_edge(a, y);
        (g, a, y)
    }

    #[test]
    fn scenario_a_frontdoor_identifiable() {
        let (g, cause, effect) = scenario_a_frontdoor();
        let result = identify(&g, &[cause], &[effect]);
        let sig = result.expect("front-door must be identifiable");
        // Signature backbone should include Y (effect).
        assert!(sig.contains(effect));
    }

    #[test]
    fn scenario_b_backdoor_identifiable() {
        let (g, cause, effect) = scenario_b_backdoor();
        let result = identify(&g, &[cause], &[effect]);
        let sig = result.expect("back-door must be identifiable");
        assert!(sig.contains(effect));
    }

    #[test]
    fn scenario_c_game_kg_identifiable_excludes_confounder_neighbor() {
        let (g, cause, effect) = scenario_c_game_kg();
        let result = identify(&g, &[cause], &[effect]);
        let sig = result.expect("game KG scenario must be identifiable");

        // Ground truth: signature should include E2 branch (the back-door
        // path) + Outcome, but NOT NPC1 (the confounder neighbor).
        let npc1 = n(5);
        let e2 = n(9);
        assert!(
            sig.contains(effect),
            "Outcome must be in signature: {sig:?}"
        );
        assert!(
            sig.contains(e2),
            "E2 (the surviving back-door branch) must be in signature: {sig:?}"
        );
        assert!(
            !sig.contains(npc1),
            "NPC1 must NOT be in signature (the do(E1) cut severs it): {sig:?}"
        );
    }

    #[test]
    fn scenario_d_bowarc_not_identifiable() {
        let (g, cause, effect) = scenario_d_bowarc();
        let result = identify(&g, &[cause], &[effect]);
        assert!(
            matches!(result, Err(IdentificationError::NotIdentifiable { .. })),
            "bow-arc must NOT be identifiable: got {result:?}"
        );
    }

    #[test]
    fn empty_cause_or_effect_is_rejected() {
        let (g, _, effect) = scenario_a_frontdoor();
        let result = identify(&g, &[], &[effect]);
        assert!(matches!(result, Err(IdentificationError::EmptyQuery)));
    }

    #[test]
    fn hedge_pair_included_in_error_when_known() {
        let (g, cause, effect) = scenario_d_bowarc();
        let result = identify(&g, &[cause], &[effect]);
        if let Err(IdentificationError::NotIdentifiable { hedge, .. }) = result {
            assert!(hedge.is_some(), "bow-arc should populate hedge pair");
        } else {
            panic!("expected NotIdentifiable");
        }
    }
}
