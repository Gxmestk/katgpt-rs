//! Pinned cross-target vectors — **the actual deliverable of Issues 108/109**.
//!
//! The code in this crate is the easy half. What keeps a device and a node
//! from silently disagreeing is this table, asserted on **both** sides: here,
//! and in every repo that owns one of the upstream implementations. A green
//! test on one side proves nothing about the other.
//!
//! ## Failure mode being defended against
//!
//! Two implementations of the rejection-sampling threshold will drift. When
//! they do, device and node compute different items from the same seed — and
//! **every honest claim looks like fraud**, indistinguishable from a player
//! cheating. The table turns that silent, un-attributable failure into a red
//! test.
//!
//! ## How these were produced
//!
//! `cargo run -p katgpt-device-verify --features std --example gen_vectors`,
//! which is committed alongside them. Seeds are `BLAKE3(label)` over short
//! ASCII labels so anyone can regenerate the table from the labels alone —
//! no opaque byte blobs whose provenance dies with the commit.
//!
//! ## Coverage rationale
//!
//! `sides` values are split deliberately:
//!
//! - **Divides 256** (`1, 2, 4, 8, 16, 32, 64, 128`) — `256 % sides == 0`, so
//!   `threshold == 256`, the first byte is *always* accepted and the fallback
//!   branch is dead. These pin the common path.
//! - **Does not divide 256** (`3, 5, 6, 7, 10, 12, 20, 100, 255`) — the
//!   rejection branch is live. **This is where drift hides**, because it is
//!   the only place where two well-meaning implementations can reasonably
//!   differ (reject-and-retry vs. reject-and-fall-back, and *which* byte the
//!   fallback reads).
//! - [`FAIR_ROLL_FALLBACK_VECTORS`] holds seeds *searched for* so that
//!   `hash[0] >= threshold` — the fallback branch is exercised by
//!   construction, not by luck. Without these the branch is hit ~0.4% of the
//!   time for `sides = 3` and a drifting implementation ships green.

use crate::Hash;

/// One pinned `(seed, sides) → die` fair-roll vector.
#[derive(Clone, Copy, Debug)]
pub struct FairRollVector {
    /// The label whose `BLAKE3` hash is [`Self::seed`]. Provenance, not input.
    pub label: &'static str,
    /// The combined `BLAKE3(α ‖ β)` seed.
    pub seed: Hash,
    /// Number of die faces.
    pub sides: u8,
    /// The expected result, in `1..=sides`.
    pub die: u8,
}

/// One pinned Merkle inclusion vector.
#[derive(Clone, Copy, Debug)]
pub struct MerkleVector {
    /// Provenance label for the leaf set.
    pub label: &'static str,
    /// The leaf hash being proven.
    pub leaf: Hash,
    /// The leaf's index in the tree.
    pub index: usize,
    /// Sibling hashes, leaf level → root level.
    pub siblings: &'static [Hash],
    /// The root the proof must reconstruct.
    pub root: Hash,
}

include!("vectors_data.rs");
