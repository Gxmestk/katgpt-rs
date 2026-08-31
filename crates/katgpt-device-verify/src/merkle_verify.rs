//! Binary-Merkle inclusion verification — the device `Curator` path.
//!
//! Bit-identical to `riir_neuron_db::merkle::{hash_pair,
//! compute_root_from_proof, verify_proof}`, which is what `riir-chain`'s
//! `catchup::merkle` re-exports and what `build_tier_root` /
//! `build_block_root` commit to.
//!
//! ## Verify-only, by design
//!
//! There is no `MerkleTree` here. Building a tree needs the leaf set; a
//! `Satellite` holds a root and a ≤640-byte proof and nothing else. Tree
//! *construction* stays in `riir-neuron-db` — one implementation, and the
//! device links the half it can afford.
//!
//! ## Cost
//!
//! `compute_root_from_proof` is one [`hash_pair`] per proof level, so a
//! 2²⁰-leaf tree costs at most 20 BLAKE3 hashes of 64 bytes. Host-measured at
//! 847 ps per root compare and 4.55 ns per zone; even a pessimistic 150×
//! Xtensa penalty leaves a full proof verification far inside a Glacial
//! (≤ 0.1 Hz) budget. The work was never the problem — the *linkability* was.

use crate::{HASH_SIZE, Hash};

/// Maximum tree depth, mirroring `riir_neuron_db::merkle::MAX_DEPTH`.
///
/// A proof at depth `D` carries `D` sibling hashes, so this bounds a proof at
/// `20 × 32 = 640` bytes — the figure that makes a device-side proof buffer a
/// fixed-size stack array instead of an allocation.
pub const MAX_DEPTH: usize = 20;

/// Maximum proof size in sibling hashes.
pub const MAX_PROOF_SIZE: usize = MAX_DEPTH;

/// Sentinel sibling for an unpaired node — an **opaque pinned constant**.
///
/// `riir-neuron-db` pairs an odd level's last node with this rather than
/// promoting it unchanged, and `MerkleTree::proof` pushes it as the sibling
/// for that position. A device verifying a proof over an odd-sized tree
/// therefore receives this value on the wire, and must agree on it byte for
/// byte or it computes a different root for every such tree.
///
/// # It is NOT `BLAKE3("")`, despite what upstream says
///
/// `riir-neuron-db/src/merkle.rs:41` documents this constant as *"BLAKE3 of
/// the empty input, pre-computed to avoid rehashing."* **That comment is
/// false.** `BLAKE3(b"")` is `af1349b9f5f9a1a6…`; this constant is
/// `afe452d4881b850d…`. They share only the first byte. No test upstream
/// asserts the preimage, so the claim has never been checked — this crate's
/// `empty_hash_is_the_pinned_sentinel` test is the first thing to check it,
/// and it failed on the first run.
///
/// The constant is still *fine as a sentinel*: an unpaired-node marker needs
/// to be fixed and agreed, not to have any particular preimage (having no
/// known preimage is arguably the stronger property). What is not fine is the
/// comment, because **a device port is written from the comment**. Anyone
/// implementing this seam from that sentence computes `BLAKE3("")`, and every
/// odd-sized tree silently disagrees — the exact "every honest claim looks
/// like fraud" failure this crate exists to prevent, one commit away from
/// shipping.
///
/// So: do **not** "correct" these bytes to `BLAKE3("")`. That would fork the
/// root of every historical odd tree. Correct the comment upstream instead
/// (`riir-neuron-db` Issue filed 2026-08-24).
pub const EMPTY_HASH: Hash = [
    0xAF, 0xE4, 0x52, 0xD4, 0x88, 0x1B, 0x85, 0x0D, 0x41, 0x9F, 0x2E, 0x3A, 0x3C, 0x73, 0xE2, 0x2F,
    0x30, 0x6B, 0x5C, 0xE9, 0x5E, 0x6B, 0xB0, 0x93, 0x5A, 0x36, 0x75, 0xF3, 0x01, 0x31, 0x1B, 0x09,
];

/// Hash a pair of 32-byte hashes with BLAKE3.
///
/// The single choke point — every hash in this module goes through here, which
/// is the same discipline `riir-neuron-db` keeps upstream. Stack-allocated
/// 64-byte buffer, one-shot `blake3::hash`; the output is identical to a
/// two-call streaming `Hasher::update` because BLAKE3 is incremental for
/// single-chunk inputs ≤ 1024 bytes.
#[inline]
#[must_use]
pub fn hash_pair(left: &Hash, right: &Hash) -> Hash {
    let mut buf = [0u8; HASH_SIZE * 2];
    buf[..HASH_SIZE].copy_from_slice(left);
    buf[HASH_SIZE..].copy_from_slice(right);
    *blake3::hash(&buf).as_bytes()
}

/// Compute the root implied by a leaf, its index, and its sibling path.
///
/// Walks leaf → root. The sibling sits on the left when the current index is
/// odd and on the right when it is even; the index is shifted right one bit
/// per level. Mirrors the Lean 4 spec `NeuronDbProof.Merkle.computeRootFromProof`.
#[inline]
#[must_use]
pub fn compute_root_from_proof(leaf_hash: &Hash, mut index: usize, siblings: &[Hash]) -> Hash {
    let mut current = *leaf_hash;
    for sibling in siblings {
        current = match index & 1 {
            0 => hash_pair(&current, sibling),
            _ => hash_pair(sibling, &current),
        };
        index >>= 1;
    }
    current
}

/// Verify an inclusion proof against an expected root.
///
/// Pure function — no tree state. This is the whole light-client surface: a
/// device that holds `(leaf, index, siblings, root)` can answer "is this leaf
/// really in that commitment?" without holding the zone set.
#[inline]
#[must_use]
pub fn verify_proof(leaf_hash: &Hash, leaf_index: usize, siblings: &[Hash], root: &Hash) -> bool {
    &compute_root_from_proof(leaf_hash, leaf_index, siblings) == root
}

/// Verify an inclusion proof, rejecting an over-deep proof instead of walking it.
///
/// A proof longer than [`MAX_PROOF_SIZE`] cannot have come from a well-formed
/// `riir-neuron-db` tree — that repo caps the tree at `2^MAX_DEPTH` leaves with
/// a hard `assert!`. On a device the length is attacker-controlled, and
/// walking an unbounded path is unbounded work in a Glacial budget. Prefer
/// this over [`verify_proof`] on anything that arrives over a wire.
#[inline]
#[must_use]
pub fn verify_proof_bounded(
    leaf_hash: &Hash,
    leaf_index: usize,
    siblings: &[Hash],
    root: &Hash,
) -> bool {
    if siblings.len() <= MAX_PROOF_SIZE {
        verify_proof(leaf_hash, leaf_index, siblings, root)
    } else {
        false
    }
}

/// Compare two roots for equality.
///
/// Named rather than inlined at call sites because "the roots differ" is the
/// Byzantine-inconsistency signal a `Curator` reports, and it should be
/// greppable. Plain `==` is correct: a Merkle root is public data, so there is
/// no secret for a timing side-channel to leak.
#[inline]
#[must_use]
pub fn roots_match(a: &Hash, b: &Hash) -> bool {
    a == b
}
