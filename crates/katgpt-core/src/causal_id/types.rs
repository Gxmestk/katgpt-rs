//! Core types for syntactic causal identification (Plan 457, Research 450).
//!
//! These types model an Acyclic Directed Mixed Graph (ADMG) — the substrate
//! the Cakiqi-Little identification algorithm (arXiv:2403.09580) operates on.
//!
//! ## Vocabulary
//!
//! - **Node** — a variable in the causal graph, referenced by BLAKE3 [`NodeId`]
//!   (so KG triples / shards / zones can be referenced by their content hash).
//! - **Directed edge** `a → b` — `a` is a direct cause of `b`.
//! - **Bidirected edge** `a ↔ b` — a latent (unobserved) confounder influences
//!   both endpoints. This is what Canvas FlowGraph reachability cannot see.
//! - **ADMG** — the mixed graph carrying both edge kinds.
//! - **AdmgSignature** — the interventional signature backbone `Y⋆ =
//!   An(Y in G[V\A])`, the set of nodes the identification algorithm had to
//!   consider to derive `Σ_{Y|do(A)}`. This is the non-trivial information
//!   that Canvas reachability cannot derive.
//!
//! ## Why modelless
//!
//! Every type here is pure data. The identification algorithm
//! ([`crate::causal_id::identify`]) is pure graph rewriting on these types —
//! no backprop, no gradient descent. The ADMG itself is constructed
//! downstream (Plan 457 Phase 3, riir-ai) from a `KgTriple` corpus +
//! a confounder injection layer.

use arrayvec::ArrayVec;

/// Maximum node count we promise the alloc-free read path can carry in a
/// fixed-size signature. The Plan 457 Phase 2 G2 perf gate benchmarks at
/// 32 nodes; signatures larger than this fall back to a heap-allocated
/// `Vec<NodeId>` via [`AdmgSignature::Heap`].
///
/// Chosen so `ArrayVec<NodeId, 32>` (32 × 32 bytes = 1 KiB) fits comfortably
/// in a stack frame without blowing the red zone.
pub const INLINE_SIGNATURE_CAP: usize = 32;

/// A node identifier in an ADMG — a 32-byte BLAKE3 content hash.
///
/// Using a content hash (instead of a `u32` ordinal) lets the same `NodeId`
/// reference a KG triple, a NeuronShard, a zone, or any other BLAKE3-committed
/// artifact in the 7-repo stack. The trade-off is 32 bytes vs 4, but `NodeId`
/// is `Copy` and `#[repr(transparent)]` over `[u8; 32]`, so it has no
/// alignment overhead.
///
/// ## Construction
///
/// - [`NodeId::from_u32`] — canonical ordinal construction (used by tests
///   and any caller that builds an ADMG procedurally with small integer
///   IDs). The ordinal lives in the low 4 bytes big-endian; the remaining
///   28 bytes are zero.
/// - [`NodeId::from_label`] — BLAKE3 hash of an arbitrary label string.
///   Use this when the node references a real artifact (KG triple, shard,
///   etc.) whose identity should be content-derived.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct NodeId(pub [u8; 32]);

impl NodeId {
    /// Construct a `NodeId` from a `u32` ordinal. The ordinal lives in the
    /// high 4 bytes big-endian; the low 28 bytes are zero. This is the
    /// canonical way to build node IDs for tests + procedural ADMG
    /// construction.
    #[inline]
    pub const fn from_u32(n: u32) -> Self {
        let mut bytes = [0u8; 32];
        bytes[0] = (n >> 24) as u8;
        bytes[1] = (n >> 16) as u8;
        bytes[2] = (n >> 8) as u8;
        bytes[3] = n as u8;
        Self(bytes)
    }

    /// Construct a `NodeId` by BLAKE3-hashing a label. Use this when the
    /// node references a content-defined artifact (KG triple, shard, zone).
    #[inline]
    pub fn from_label(label: &[u8]) -> Self {
        Self(*blake3::hash(label).as_bytes())
    }

    /// Recover the ordinal if this `NodeId` was constructed via
    /// [`NodeId::from_u32`]. Returns `None` for BLAKE3-derived IDs whose
    /// low 28 bytes are non-zero. Used by [`Display`](core::fmt::Display)
    /// to render ordinal IDs as `N<n>`.
    #[inline]
    pub const fn as_u32(&self) -> Option<u32> {
        let b = &self.0;
        // Manual "any" over the low 28 bytes — const fn cannot use iterators.
        let mut i = 4;
        while i < 32 {
            if b[i] != 0 {
                return None;
            }
            i += 1;
        }
        Some(((b[0] as u32) << 24) | ((b[1] as u32) << 16) | ((b[2] as u32) << 8) | (b[3] as u32))
    }
}

impl core::fmt::Debug for NodeId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.as_u32() {
            Some(n) => write!(f, "NodeId(N{n})"),
            None => write!(f, "NodeId({})", HexPrefix(&self.0[..4])),
        }
    }
}

impl core::fmt::Display for NodeId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.as_u32() {
            Some(n) => write!(f, "N{n}"),
            None => write!(f, "{}", HexPrefix(&self.0[..4])),
        }
    }
}

/// Wrapper to hex-format a byte slice as a `Display` target.
struct HexPrefix<'a>(&'a [u8]);

impl<'a> core::fmt::Display for HexPrefix<'a> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        use core::fmt::Write as _;
        const HEX: &[u8; 16] = b"0123456789abcdef";
        for &b in self.0 {
            f.write_char(HEX[(b >> 4) as usize] as char)?;
            f.write_char(HEX[(b & 0x0f) as usize] as char)?;
        }
        f.write_char('…')
    }
}

/// An Acyclic Directed Mixed Graph — directed edges + bidirected edges over
/// a set of nodes.
///
/// Directed edges `a → b` mean `a` is a direct cause of `b`. Bidirected edges
/// `a ↔ b` mean a latent (unobserved) confounder influences both endpoints.
/// Bidirected edges are stored canonically as `(min, max)` so dedup and
/// equality comparisons are straightforward.
///
/// **Acyclicity** is the caller's responsibility — the identification
/// algorithm assumes it but does not verify it. The kg→ADMG bridge in
/// Plan 457 Phase 3 will enforce acyclicity at construction time.
#[derive(Clone, Debug, Default)]
pub struct Admg {
    /// Nodes in the graph. Order is not semantically meaningful but is
    /// preserved by [`Self::subgraph`] / [`Self::fix_node`] for determinism.
    pub nodes: Vec<NodeId>,
    /// Directed edges `(parent, child)` — `parent → child`.
    pub directed: Vec<(NodeId, NodeId)>,
    /// Bidirected edges `(a, b)` with `a <= b` — represents a latent
    /// confounder influencing both endpoints.
    pub bidirected: Vec<(NodeId, NodeId)>,
}

impl Admg {
    /// Construct an empty ADMG over the given node set (no edges).
    pub fn new(nodes: Vec<NodeId>) -> Self {
        Self {
            nodes,
            directed: Vec::new(),
            bidirected: Vec::new(),
        }
    }

    /// Add a directed edge `parent → child`. Builder-style.
    pub fn directed_edge(&mut self, parent: NodeId, child: NodeId) -> &mut Self {
        self.directed.push((parent, child));
        self
    }

    /// Add a bidirected edge `a ↔ b` (latent confounder). Stored as
    /// `(min(a,b), max(a,b))` so dedup is by canonical ordering.
    /// Builder-style.
    pub fn bidirected_edge(&mut self, a: NodeId, b: NodeId) -> &mut Self {
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        self.bidirected.push((lo, hi));
        self
    }

    /// Number of nodes in the graph.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// True iff `v` is in the node set.
    pub fn contains_node(&self, v: NodeId) -> bool {
        self.nodes.contains(&v)
    }
}

/// The interventional signature backbone — `Y⋆ = An(Y in G[V\A])`, the set
/// of nodes the identification algorithm had to consider to derive
/// `Σ_{Y|do(A)}`.
///
/// Per Cakiqi-Little Theorem 1 the strict signature objects are `Y` itself
/// (the `Hide_{Y⋆\Y}` operation removes the rest from the output). But the
/// *derivation structure* references all of `Y⋆`, so we expose the full
/// backbone — that is what makes this primitive's answer richer than
/// Canvas reachability's boolean.
///
/// ## Inline vs heap
///
/// Signatures of `<= INLINE_SIGNATURE_CAP` (32) nodes are stored inline in
/// an `ArrayVec` — no heap allocation on the read path. Larger signatures
/// fall back to a heap `Vec`. The 32-node threshold matches the Plan 457
/// Phase 2 G2 perf gate target.
///
/// ## clippy note
///
/// The `Inline` variant is deliberately large (32 × 32 = 1024 bytes) so it
/// can carry a full 32-node signature without indirection. Boxing would
/// defeat the alloc-free read path. The variant-size disparity with `Heap`
/// is a measured trade-off, not an oversight.
#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum AdmgSignature {
    /// Inline signature — no heap allocation. Used when the signature fits
    /// in `INLINE_SIGNATURE_CAP` (32) nodes.
    Inline(ArrayVec<NodeId, INLINE_SIGNATURE_CAP>),
    /// Heap-allocated signature — used when the signature exceeds the inline
    /// capacity. The read path still allocates only once per `identify()`
    /// call; the inner recursion does not allocate.
    Heap(Vec<NodeId>),
}

impl AdmgSignature {
    /// Construct a signature from an iterator of nodes. Uses the inline
    /// variant if the iterator yields ≤ `INLINE_SIGNATURE_CAP` nodes,
    /// otherwise spills to heap.
    pub fn from_nodes<I: IntoIterator<Item = NodeId>>(iter: I) -> Self {
        let mut inline: ArrayVec<NodeId, INLINE_SIGNATURE_CAP> = ArrayVec::new();
        let mut heap: Option<Vec<NodeId>> = None;
        for n in iter {
            match &mut heap {
                Some(h) => h.push(n),
                None => {
                    if inline.is_full() {
                        let mut h = Vec::with_capacity(inline.len() + 1);
                        h.extend(inline.drain(..));
                        h.push(n);
                        heap = Some(h);
                    } else {
                        inline.push(n);
                    }
                }
            }
        }
        match heap {
            Some(h) => Self::Heap(h),
            None => Self::Inline(inline),
        }
    }

    /// Construct an empty signature.
    pub fn empty() -> Self {
        Self::Inline(ArrayVec::new())
    }

    /// Number of nodes in the signature.
    pub fn len(&self) -> usize {
        match self {
            Self::Inline(v) => v.len(),
            Self::Heap(v) => v.len(),
        }
    }

    /// True iff the signature is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate over the nodes in the signature.
    pub fn iter(&self) -> impl Iterator<Item = NodeId> + '_ {
        match self {
            Self::Inline(v) => v.iter().copied(),
            Self::Heap(v) => v.iter().copied(),
        }
    }

    /// Does the signature contain `v`?
    pub fn contains(&self, v: NodeId) -> bool {
        match self {
            Self::Inline(a) => a.contains(&v),
            Self::Heap(a) => a.contains(&v),
        }
    }

    /// Const-true iff the signature is inline (no heap allocation).
    pub fn is_inline(&self) -> bool {
        matches!(self, Self::Inline(_))
    }
}

impl PartialEq for AdmgSignature {
    fn eq(&self, other: &Self) -> bool {
        let len = self.len();
        if len != other.len() {
            return false;
        }
        // Order-insensitive equality: O(n²) but n is bounded by
        // INLINE_SIGNATURE_CAP for the hot path.
        self.iter().all(|n| other.contains(n))
    }
}

impl Eq for AdmgSignature {}

/// Identification failure modes.
///
/// The `NotIdentifiable` variant carries the hedge pair `(a, b)` so the GM
/// tool can explain WHY the query isn't identifiable — better UX than a bare
/// error (Plan 457 key design decision #4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IdentificationError {
    /// The graph contains a hedge — the interventional query has no valid
    /// syntactic derivation. The canonical case is the bow-arc
    /// (`A → Y` + `A ↔ Y` on the same pair).
    ///
    /// `cause`/`effect` echo the query for downstream diagnostics. If the
    /// hedge pair can be identified at fail time, it is provided; otherwise
    /// the hedge is unknown and only the query is echoed.
    NotIdentifiable {
        /// Echo of the original cause set head (for diagnostics). Multiple
        /// causes are not collapsed here — the caller already knows the
        /// query.
        cause: NodeId,
        /// Echo of the original effect.
        effect: NodeId,
        /// The hedge pair `(a, b)` if known. `None` for abstract hedge
        /// failures (e.g. district-of-V containment of effect).
        hedge: Option<(NodeId, NodeId)>,
    },
    /// A fixing sequence could not be completed (every greedy ordering got
    /// stuck). Treated identically to `NotIdentifiable` from the caller's
    /// perspective but kept distinct for diagnostics.
    FixFailed { cause: NodeId, effect: NodeId },
    /// The query was empty — `identify()` was called with no cause nodes
    /// and no effect nodes, or with disjoint / out-of-graph node sets.
    EmptyQuery,
}

impl core::fmt::Display for IdentificationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotIdentifiable {
                cause,
                effect,
                hedge,
            } => match hedge {
                Some((a, b)) => write!(
                    f,
                    "not identifiable: hedge ({a}, {b}) blocks Σ_{{{effect}|do({cause})}}"
                ),
                None => write!(
                    f,
                    "not identifiable: Σ_{{{effect}|do({cause})}} has no derivation"
                ),
            },
            Self::FixFailed { cause, effect } => {
                write!(f, "fix sequence failed for Σ_{{{effect}|do({cause})}}")
            }
            Self::EmptyQuery => write!(f, "empty query: cause or effect set is empty"),
        }
    }
}

impl core::error::Error for IdentificationError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_id_u32_roundtrip() {
        for n in [0u32, 1, 42, 0xffff_ffff, 0x1234_5678] {
            let id = NodeId::from_u32(n);
            assert_eq!(id.as_u32(), Some(n), "roundtrip {n}");
        }
    }

    #[test]
    fn node_id_label_is_nonzero() {
        let id = NodeId::from_label(b"npc1");
        assert_eq!(id.as_u32(), None, "BLAKE3-derived IDs aren't u32");
        // Sanity: at least one byte set in the low 28 bytes.
        assert!(id.0.iter().skip(4).any(|&b| b != 0));
    }

    #[test]
    fn node_id_display_ordinal_vs_hash() {
        let ordinal = NodeId::from_u32(7);
        assert_eq!(ordinal.to_string(), "N7");
        let hashed = NodeId::from_label(b"abc");
        let s = hashed.to_string();
        assert!(s.ends_with('…'));
        // 4 bytes hex = 8 chars + 1 ellipsis char = 9 chars total
        // (note: the ellipsis char is 3 bytes UTF-8, but `.chars().count()` is 9).
        assert_eq!(s.chars().count(), 8 + 1);
    }

    #[test]
    fn node_id_ordering_by_ordinal() {
        let a = NodeId::from_u32(0);
        let b = NodeId::from_u32(1);
        assert!(a < b);
    }

    #[test]
    fn signature_inline_below_cap_heap_above() {
        let small: Vec<NodeId> = (0..INLINE_SIGNATURE_CAP as u32)
            .map(NodeId::from_u32)
            .collect();
        let sig = AdmgSignature::from_nodes(small.iter().copied());
        assert!(sig.is_inline());
        assert_eq!(sig.len(), INLINE_SIGNATURE_CAP);

        let big: Vec<NodeId> = (0..(INLINE_SIGNATURE_CAP + 1) as u32)
            .map(NodeId::from_u32)
            .collect();
        let sig_big = AdmgSignature::from_nodes(big.iter().copied());
        assert!(!sig_big.is_inline());
        assert_eq!(sig_big.len(), INLINE_SIGNATURE_CAP + 1);
    }

    #[test]
    fn signature_equality_is_order_insensitive() {
        let a = AdmgSignature::from_nodes([NodeId::from_u32(0), NodeId::from_u32(1)]);
        let b = AdmgSignature::from_nodes([NodeId::from_u32(1), NodeId::from_u32(0)]);
        assert_eq!(a, b);
    }

    #[test]
    fn error_display_includes_hedge_when_known() {
        let err = IdentificationError::NotIdentifiable {
            cause: NodeId::from_u32(0),
            effect: NodeId::from_u32(1),
            hedge: Some((NodeId::from_u32(0), NodeId::from_u32(1))),
        };
        let s = format!("{err}");
        assert!(s.contains("hedge"));
        assert!(s.contains("N0"));
        assert!(s.contains("N1"));
    }
}
