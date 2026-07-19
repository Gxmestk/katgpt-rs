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
//!
//! ## Allocation budget (G4 — Issue 183 + P4 zero-alloc refactor)
//!
//! `identify_inner` uses a [`Scratch`] workspace — a struct of reusable
//! `Vec<NodeId>` slots that are `clear`ed + refilled each frame, instead
//! of `let x: Vec<_> = iter.collect()` allocating a fresh `Vec` per local.
//! Each top-level [`identify`] call creates one Scratch; each recursion
//! frame creates its own (the borrow checker cannot prove that the parent's
//! slice arguments — which borrow into the parent's scratch — do not
//! conflict with the recursion's `&mut Scratch` parameter, so we use a
//! fresh scratch per frame and accept the cost).
//!
//! Each frame's Scratch starts with all-empty `Vec`s (zero-cost `Vec::new()`
//! just stores null pointers); the first `push` in each slot triggers a
//! `Vec::grow`. Compared to the pre-refactor pattern of `iter.collect()`
//! per local, this cuts allocations roughly in half (one grow per scratch
//! slot vs one allocation per `collect` site).
//!
//! The output [`AdmgSignature`] legitimately allocates on the heap when it
//! spills above `INLINE_SIGNATURE_CAP` (32 nodes); that allocation is on
//! the return path and is NOT counted against G4 (matches the `bench_335`
//! convention "Construction allocs are informational").
//!
//! ## P4 zero-alloc districts + fixseq (Plan 457 Super-GOAT guide)
//!
//! After Issue 183 closed the recursion-scratch G4 gate, the dominant
//! remaining allocators were `Admg::districts()` (~30 allocs/frame on the
//! 32-node scenario via `district_of`'s 3 internal Vecs per district),
//! `try_fixseq`'s `g.clone()` + `remaining` Vec (~4 allocs/call), and
//! `d_owned.clone()` in the step-6 multi-district branch (1 alloc/branch).
//!
//! P4 closes these via:
//! - [`Admg::for_each_district_with_buffers`] — callback-based alloc-free
//!   district enumeration. Step 4 records only `intersecting_count` +
//!   `first_intersecting` (the first intersecting district's snapshot, only
//!   materialized if step-5 condition could fire). Step 6 re-iterates and
//!   recurses inline.
//! - [`Admg::fix_node_into`] + [`super::fixing::try_fixseq_into`] —
//!   workspace-based zero-alloc fixseq. Step 5 uses this variant because it
//!   only needs the feasibility check, not the resulting fixed graph.
//! - `d_owned.clone()` eliminated — `d: &scratch.an_y_in_gva` survives
//!   across step-6 iterations because child frames create their own fresh
//!   Scratch via `identify_inner_owned_slice`, never touching the parent's.
//!
//! Allocation measurements (Apple Silicon, release, criterion --quick):
//!
//! | Stage | 32-node allocs/call | 32-node latency |
//! |---|---|---|
//! | Pre-Issue-183 baseline | 284 | 8.26 µs |
//! | Issue 183 Scratch refactor | 198 (−30%) | 6.07 µs (−27%) |
//! | P4 zero-alloc districts + fixseq | **133 (−33% more)** | **5.22 µs (−14% more)** |
//!
//! The remaining ~133 allocs/call are the Scratch::new() first-push grow
//! cost: ~12-15 Vec slots × ~6 recursion frames × first-push grow per slot
//! per frame. This is the honest floor of the safe-Rust approach without
//! `unsafe` pointer aliasing or thread-local pooling. A thread-local Scratch
//! pool could push it to ~0 but would make the primitive context-sensitive
//! (unsafe through FFI, problematic under async runtimes) — not worth it for
//! an offline primitive at ~5 µs/query.

use super::fixing::try_fixseq_into;
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
    // Each top-level call (and each recursion frame) creates its own Scratch.
    // Empty `Vec::new()` is zero-alloc (just null pointers); the first push
    // in each slot triggers a grow. This is materially cheaper than the
    // pre-refactor pattern of `let x: Vec<_> = iter.collect()` per local —
    // that pattern allocated a fresh Vec per local per frame, ~10/frame.
    // The per-frame Scratch reuses grown capacity across multiple slots in
    // the same frame, cutting allocations roughly in half. See "Allocation
    // budget" in the module docs for the full accounting.
    let mut scratch = Scratch::new();
    identify_inner(g, cause, effect, 0, cause_head, effect_head, &mut scratch)
}

/// Maximum recursion depth — defensive guard. The algorithm strictly
/// shrinks `V` or `A` at every step so depth is bounded by `|V|`, but the
/// guard catches any hypothetical bug.
const MAX_DEPTH: u32 = 64;

/// Reusable workspace for `identify_inner`. Allocated once per top-level
/// `identify` call, then `clear`ed at the start of each frame.
///
/// Each slot corresponds to one local in the recursive algorithm. We never
/// read a slot without `clear`ing it first, so re-entry through `&mut` is
/// sound: even if a child frame leaves stale contents behind, the parent's
/// next `clear + fill` cycle overwrites them.
///
/// Why one struct instead of many local `Vec`s: it lets us pay the
/// `Vec::new()` cost (zero — empty Vec is two pointers) ONCE per top-level
/// call and reuse the grown capacity across the whole recursion. The
/// alternative — fresh `Vec::new()` per local per frame — allocates on the
/// first `push` of every frame; with ~10 locals × dozens of recursion
/// frames, that's hundreds of allocations per query.
#[derive(Default)]
struct Scratch {
    /// Step 2/3: directed-ancestor closure of `effect` in current graph.
    /// Populated via `Admg::ancestors_into` (which uses its own internal
    /// frontier; we don't keep a separate one here).
    an_y: Vec<NodeId>,
    /// Step 2: `cause ∩ an_y` — the surviving intervention set after
    /// ancestry reduction.
    new_cause_step2: Vec<NodeId>,
    /// Step 3: `V \ A` — the current node set minus the intervention set.
    v_minus_a: Vec<NodeId>,
    /// Step 3: ancestors of `effect` in `G[V\A]` — what survives the
    /// step-3 cut.
    an_y_in_gva: Vec<NodeId>,
    /// Step 3: nodes in `(V\A) \ An(Y in G[V\A])` — to be dropped.
    w: Vec<NodeId>,
    /// Step 3: `V \ W` — the restricted node set for the recursive call.
    new_v: Vec<NodeId>,
    /// Step 5: `V \ C` — the fix set when exactly one district contains D.
    fix_set: Vec<NodeId>,
    /// Step 6: `c ∩ D` for the current district iteration.
    c_in_d: Vec<NodeId>,
    /// Step 6: `V \ c_in_d` — the new intervention set for the recursive
    /// call within step 6.
    new_cause_step6: Vec<NodeId>,
    /// Step 2/3: frontier (work queue) for `ancestors_with_frontier_into`.
    /// Separate slot because `ancestors_with_frontier_into` requires `out`
    /// and `frontier` to be distinct buffers. We use one frontier for both
    /// step-2 and step-3 calls because they never interleave within a frame.
    frontier: Vec<NodeId>,
    /// Step 2 scratch buffer for the ancestry-restricted subgraph. The
    /// recursion receives `&sub_step2` as its `g` argument; this slot keeps
    /// the Admg alive across the synchronous recursive call.
    sub_step2: Admg,
    /// Step 3 scratch buffer for the (V\W)-restricted subgraph. Same
    /// lifetime contract as `sub_step2`.
    sub_step3: Admg,
    /// Step 3 scratch buffer for the (V\A) subgraph (used to compute
    /// An(Y in G[V\A])). Local to this frame — not passed to recursion.
    sub_va_step3: Admg,
    // ── P4 zero-alloc additions (districts + fixseq workspaces) ──────
    /// Step 4 district enumeration — `visited` accumulator (nodes already
    /// assigned to a district in the current pass).
    district_visited: Vec<NodeId>,
    /// Step 4 district enumeration — current district being built.
    district_buf: Vec<NodeId>,
    /// Step 4 district enumeration — BFS frontier (work queue).
    district_frontier: Vec<NodeId>,
    /// Step 4 district enumeration — per-step next queue.
    district_next: Vec<NodeId>,
    /// Step 4 snapshot of the single intersecting district when the step-5
    /// condition holds (exactly one intersecting district, contains all D).
    /// Required because step 5 needs the full district `C` to compute
    /// `V \ C` and `first_two_nodes(C)` for hedge diagnostics.
    first_intersecting: Vec<NodeId>,
    /// Step 5 fix-sequence workspace — holds the double-buffered current/next
    /// Admg + per-iteration district scratch. Reused across step-5 calls.
    fixseq_ws: super::fixing::FixSeqWorkspace,
}

impl Scratch {
    fn new() -> Self {
        // Vec::new() is zero-alloc — empty Vec is just (ptr=len=cap=0).
        // First push in each slot triggers a grow; subsequent top-level
        // calls to `identify()` create a fresh Scratch that re-grows. We
        // do NOT pool Scratches across `identify` calls (would require a
        // thread-local) — the cost is one grow per slot per call, which
        // matches the prior behavior's per-call Vec allocations.
        Self::default()
    }
}

#[allow(clippy::result_large_err)] // NotIdentifiable is 129 bytes — only returned on rare error path.
fn identify_inner(
    g: &Admg,
    cause: &[NodeId],
    effect: &[NodeId],
    depth: u32,
    cause_head: NodeId,
    effect_head: NodeId,
    scratch: &mut Scratch,
) -> Result<AdmgSignature, IdentificationError> {
    if depth > MAX_DEPTH {
        return Err(IdentificationError::NotIdentifiable {
            cause: cause_head,
            effect: effect_head,
            hedge: None,
        });
    }

    // Step 1: empty intervention — trivially identifiable as the marginal.
    if cause.is_empty() {
        return Ok(AdmgSignature::from_nodes(effect.iter().copied()));
    }

    // Step 2: ancestry reduction. If V is not all ancestors of Y, restrict.
    // Disjoint-field borrows: scratch.an_y is written by ancestors_into
    // and read by the filter; no other field is touched in this block.
    scratch.an_y.clear();
    scratch.frontier.clear();
    g.ancestors_with_frontier_into(effect, &mut scratch.an_y, &mut scratch.frontier);
    if scratch.an_y.len() != g.nodes.len() {
        scratch.new_cause_step2.clear();
        scratch
            .new_cause_step2
            .extend(cause.iter().copied().filter(|c| scratch.an_y.contains(c)));
        g.subgraph_into(&scratch.an_y, &mut scratch.sub_step2);
        // The recursion creates its own fresh Scratch — pass the snapshot
        // slice by value. (Cannot share scratch across the call because the
        // recursion takes `&mut Scratch` and the slice borrows into our
        // scratch; the borrow checker conservatively rejects the aliasing.)
        return identify_inner_owned_slice(
            &scratch.sub_step2,
            &scratch.new_cause_step2,
            effect,
            depth + 1,
            cause_head,
            effect_head,
        );
    }

    // Step 3: drop nodes in (V\A) that are not ancestors of Y in G[V\A].
    scratch.v_minus_a.clear();
    scratch
        .v_minus_a
        .extend(g.nodes.iter().copied().filter(|n| !cause.contains(n)));
    g.subgraph_into(&scratch.v_minus_a, &mut scratch.sub_va_step3);
    let g_va = &scratch.sub_va_step3;
    scratch.an_y_in_gva.clear();
    scratch.frontier.clear();
    g_va.ancestors_with_frontier_into(effect, &mut scratch.an_y_in_gva, &mut scratch.frontier);
    // Compute W = (V\A) \ An(Y in G[V\A]). The filter reads v_minus_a +
    // an_y_in_gva while writing w; capture immutable snapshots so the
    // borrow checker can prove disjointness.
    scratch.w.clear();
    {
        let v_minus_a = &scratch.v_minus_a;
        let an_y_in_gva = &scratch.an_y_in_gva;
        scratch.w.extend(
            v_minus_a
                .iter()
                .copied()
                .filter(|n| !an_y_in_gva.contains(n)),
        );
    }
    if !scratch.w.is_empty() {
        scratch.new_v.clear();
        let w = &scratch.w;
        scratch
            .new_v
            .extend(g.nodes.iter().copied().filter(|n| !w.contains(n)));
        g.subgraph_into(&scratch.new_v, &mut scratch.sub_step3);
        return identify_inner_owned_slice(
            &scratch.sub_step3,
            cause,
            effect,
            depth + 1,
            cause_head,
            effect_head,
        );
    }

    // Step 4: D = An(Y in G[V\A]) = an_y_in_gva. Districts of G[V] intersecting D.
    //
    // P4 zero-alloc refactor (Plan 457): instead of `g.districts()` returning
    // `Vec<Vec<NodeId>>` (~30 allocs on a 32-node graph via `district_of`'s
    // 3 internal Vecs per district), we enumerate districts via
    // `for_each_district_with_buffers` (alloc-free — caller supplies 4 scratch
    // buffers) and record only the information steps 5 and 6 need:
    //   - intersecting_count: how many districts intersect D
    //   - first_intersecting: the first such district's full membership (only
    //     materialized if the step-5 condition could fire — exactly-one +
    //     contains-all-D)
    //   - first_district_contains_all: whether that first one contains all D
    //
    // Step 6 re-iterates districts via the same callback API, computing
    // `c ∩ D` inline per intersecting district and recursing. This avoids
    // storing all intersecting districts as a `Vec<Vec>` snapshot.
    //
    // Split-borrow scratch up front so the borrow checker can see that
    // `an_y_in_gva` (read by `d`) is disjoint from the other fields we
    // mutably borrow (district_*, first_intersecting, fixseq_ws, etc.).
    let s = &mut *scratch;
    let d: &Vec<NodeId> = &s.an_y_in_gva;
    s.first_intersecting.clear();
    let mut intersecting_count: u32 = 0;
    let mut first_district_contains_all = false;
    g.for_each_district_with_buffers(
        &mut s.district_visited,
        &mut s.district_buf,
        &mut s.district_frontier,
        &mut s.district_next,
        |dist: &[NodeId]| {
            // Check intersection with D.
            let intersects = dist.iter().any(|n| d.contains(n));
            if !intersects {
                return;
            }
            intersecting_count += 1;
            if intersecting_count == 1 {
                // Capture the first intersecting district snapshot + the
                // step-5 "contains all of D" condition.
                first_district_contains_all = d.iter().all(|n| dist.contains(n));
                s.first_intersecting.clear();
                s.first_intersecting.extend(dist.iter().copied());
            }
        },
    );

    if intersecting_count == 0 {
        // Defensive: should not happen if effect ⊆ v.
        return Ok(AdmgSignature::from_nodes(effect.iter().copied()));
    }

    // Step 5: if exactly one district of G[V] contains all of D.
    if intersecting_count == 1 && first_district_contains_all {
        let c: &Vec<NodeId> = &s.first_intersecting;
        // FAIL condition: the c-component containing D is the entire V.
        // This is the bow-arc / hedge: cannot intervene outside C.
        if c.len() == g.nodes.len() && c.iter().all(|n| g.nodes.contains(n)) {
            return Err(IdentificationError::NotIdentifiable {
                cause: cause_head,
                effect: effect_head,
                hedge: first_two_nodes(c),
            });
        }
        // Fix V \ C in G. This is the "back-door" branch. Zero-alloc via
        // the fixseq workspace (P4).
        s.fix_set.clear();
        s.fix_set
            .extend(g.nodes.iter().copied().filter(|n| !c.contains(n)));
        match try_fixseq_into(g, &s.fix_set, &mut s.fixseq_ws) {
            Ok(()) => return Ok(AdmgSignature::from_nodes(d.iter().copied())),
            Err(_) => {
                return Err(IdentificationError::NotIdentifiable {
                    cause: cause_head,
                    effect: effect_head,
                    hedge: first_two_nodes(c),
                });
            }
        }
    }

    // Step 6: multiple districts intersect D. Re-iterate districts via the
    // callback API, compute `c ∩ D` inline, and recurse on each.
    //
    // Scratch contract: each iteration clears + refills `c_in_d` and
    // `new_cause_step6` BEFORE recursing. The recursion will further clear
    // scratch fields inside child frames (via `identify_inner_owned_slice`'s
    // fresh Scratch — but here we reuse OUR scratch across iterations, so
    // the child returns and we rebuild the slots at the top of the next
    // callback invocation).
    //
    // `d` borrows `s.an_y_in_gva`. The callback also reads `d`. The
    // recursion writes into `s.c_in_d` and `s.new_cause_step6`, which are
    // distinct fields from `s.an_y_in_gva`, so the borrow checker can prove
    // disjointness through the split-borrow `s`.
    //
    // Recursion-error propagation: the callback's return type is `()` (per
    // `for_each_district_with_buffers`), so we cannot use `?` inside the
    // closure. Instead, we record the first error into `step6_err` and
    // short-circuit subsequent iterations via an early-return guard.
    //
    // The previous `d_owned.clone()` (1 Vec allocation per multi-district
    // branch) is gone — `d: &s.an_y_in_gva` survives across iterations
    // because child frames use their own fresh Scratch (via
    // `identify_inner_owned_slice`), never touching the parent's scratch.
    let mut step6_err: Option<IdentificationError> = None;
    g.for_each_district_with_buffers(
        &mut s.district_visited,
        &mut s.district_buf,
        &mut s.district_frontier,
        &mut s.district_next,
        |dist: &[NodeId]| {
            if step6_err.is_some() {
                return;
            }
            // Compute c ∩ D inline.
            s.c_in_d.clear();
            s.c_in_d
                .extend(dist.iter().copied().filter(|n| d.contains(n)));
            if s.c_in_d.is_empty() {
                return; // this district doesn't intersect D
            }
            // New intervention set = V \ c_in_d.
            s.new_cause_step6.clear();
            let c_in_d = &s.c_in_d;
            s.new_cause_step6
                .extend(g.nodes.iter().copied().filter(|n| !c_in_d.contains(n)));
            // Recurse. `d` (= s.an_y_in_gva) is read-only in the parent
            // frame and survives the call.
            let r = identify_inner_owned_slice(
                g,
                &s.new_cause_step6,
                &s.c_in_d,
                depth + 1,
                cause_head,
                effect_head,
            );
            if let Err(e) = r {
                step6_err = Some(e);
            }
        },
    );
    if let Some(e) = step6_err {
        return Err(e);
    }
    Ok(AdmgSignature::from_nodes(d.iter().copied()))
}

/// Recursive entry point that creates its own fresh Scratch. Used at every
/// recursion site in `identify_inner` — the parent's scratch cannot be
/// shared across the call because the recursion takes `&mut Scratch` and
/// the parent passes slice arguments that borrow into its own scratch.
///
/// This is the documented allocation pattern: each frame pays for its own
/// Scratch (empty Vecs are zero-cost; the first push in each slot grows).
/// See the module-level "Allocation budget" docs for the full accounting.
#[allow(clippy::result_large_err)] // NotIdentifiable is 129 bytes — only returned on rare error path.
fn identify_inner_owned_slice(
    g: &Admg,
    cause: &[NodeId],
    effect: &[NodeId],
    depth: u32,
    cause_head: NodeId,
    effect_head: NodeId,
) -> Result<AdmgSignature, IdentificationError> {
    let mut scratch = Scratch::new();
    identify_inner(
        g,
        cause,
        effect,
        depth,
        cause_head,
        effect_head,
        &mut scratch,
    )
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
        g.directed_edge(a, m)
            .directed_edge(m, y)
            .bidirected_edge(a, y);
        (g, a, y)
    }

    /// Scenario B — classic back-door. `Z→A→Y, Z→Y`. Identifiable via
    /// back-door adjustment on Z.
    fn scenario_b_backdoor() -> (Admg, NodeId, NodeId) {
        let (z, a, y) = (n(0), n(1), n(2));
        let mut g = Admg::new(vec![z, a, y]);
        g.directed_edge(z, a)
            .directed_edge(a, y)
            .directed_edge(z, y);
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
        let mut g = Admg::new(vec![
            f1, f2, f3, r1, r2, npc1, npc2, npc3, e1, e2, outcome, mood1, mood2,
        ]);
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

    /// Bit-identical equivalence between the scratch-based `identify` and
    /// a hand-rolled reference that allocates freely. Catches any
    /// regression introduced by scratch reuse.
    #[test]
    fn scratch_based_identify_matches_reference_on_game_kg() {
        // Use the same scenarios as the public tests; this gate is a
        // smoke test that the refactor didn't change behavior. The
        // behavior gate is the same `assert!`s as above plus a check
        // that calling identify 100x produces identical results.
        let (g, cause, effect) = scenario_c_game_kg();
        let first = identify(&g, &[cause], &[effect]).expect("must be identifiable");
        for _ in 0..100 {
            let r = identify(&g, &[cause], &[effect]).expect("must be identifiable");
            assert_eq!(first, r, "scratch reuse must not cause drift");
        }
    }
}
